import torch
import torch.nn as nn
import random

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        
        # LSTM input: word embedding
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Layers to initialize hidden and cell states from fused features (1024 dims)
        self.init_h = nn.Linear(1024, hidden_dim)
        self.init_c = nn.Linear(1024, hidden_dim)
        
    def forward(self, fused_features, target_ids=None, max_len=10, teacher_forcing_ratio=0.5):
        batch_size = fused_features.size(0)
        device = fused_features.device
        
        # Initialize hidden state from fused_features
        h = self.init_h(fused_features).unsqueeze(0) # [num_layers, batch, hidden_dim]
        c = self.init_c(fused_features).unsqueeze(0)
        
        # Start token (Assuming 0 for PhoBERT/Skeleton, adjust based on actual tokenizer)
        input_id = torch.zeros(batch_size, 1).long().to(device)
        
        outputs = []
        for t in range(max_len):
            embedded = self.embedding(input_id) # [batch, 1, input_dim]
            output, (h, c) = self.lstm(embedded, (h, c))
            logits = self.fc(output.squeeze(1)) # [batch, vocab_size]
            outputs.append(logits)
            
            if target_ids is not None and t < target_ids.size(1) and random.random() < teacher_forcing_ratio:
                input_id = target_ids[:, t].unsqueeze(1)
            else:
                input_id = logits.argmax(1).unsqueeze(1)
                
        return torch.stack(outputs, dim=1)

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, nhead=8, num_layers=2):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, hidden_dim)) # Simple positional encoding
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Map fused_features to hidden_dim for memory input
        self.memory_proj = nn.Linear(1024, hidden_dim)
        
    def forward(self, fused_features, target_ids=None, max_len=10):
        batch_size = fused_features.size(0)
        device = fused_features.device
        
        # Memory is the fused image+text features
        memory = self.memory_proj(fused_features).unsqueeze(1) # [batch, 1, hidden_dim]
        
        if target_ids is not None:
            # Training with teacher forcing (standard Transformer)
            tgt = self.embedding(target_ids) + self.pos_encoder[:, :target_ids.size(1), :]
            output = self.transformer_decoder(tgt, memory)
            return self.fc(output)
        else:
            # Inference (Greedy)
            curr_input = torch.zeros(batch_size, 1).long().to(device)
            outputs = []
            for t in range(max_len):
                tgt = self.embedding(curr_input) + self.pos_encoder[:, :curr_input.size(1), :]
                out = self.transformer_decoder(tgt, memory)
                logits = self.fc(out[:, -1, :]) # Take last token
                next_token = logits.argmax(1).unsqueeze(1)
                curr_input = torch.cat([curr_input, next_token], dim=1)
                outputs.append(logits)
            return torch.stack(outputs, dim=1)
