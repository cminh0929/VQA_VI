import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_answer_length=10, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_answer_length = max_answer_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Stability: Projection from fused features to LSTM hidden state
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, fused_features, target_ids=None):
        # fused_features: [batch, embed_size]
        batch_size = fused_features.size(0)
        
        # Correct Initialization
        h0 = self.init_h(fused_features).unsqueeze(0) # [num_layers, batch, hidden_size]
        c0 = self.init_c(fused_features).unsqueeze(0)
        
        if target_ids is not None:
            # Training: Standard Teacher Forcing
            embeddings = self.embedding(target_ids)
            outputs, _ = self.lstm(embeddings, (h0, c0))
            return self.fc(outputs)
        else:
            # Inference: Greedy with BOS handling
            device = fused_features.device
            max_len = self.max_answer_length
            # Note: PhoBERT <s> is index 0
            curr_token = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            h, c = h0, c0
            
            all_logits = []
            for _ in range(max_len):
                emb = self.embedding(curr_token)
                out, (h, c) = self.lstm(emb, (h, c))
                logits = self.fc(out)
                all_logits.append(logits)
                curr_token = logits.argmax(-1)
                
            return torch.cat(all_logits, dim=1)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, max_answer_length=10, num_heads=8, num_layers=3):
        super().__init__()
        self.max_answer_length = max_answer_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Standard Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            batch_first=True,
            dim_feedforward=embed_size * 4,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, memory, target_ids=None):
        """
        memory: [batch, seq_len, embed_size] - Sequence of visual+textual features
        """
        batch_size = memory.size(0)
        device = memory.device
        
        if target_ids is not None:
            tgt = self.embedding(target_ids)
            # Create causal mask for target
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=device)
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            return self.fc(output)
        else:
            # Robust Greedy decoding - collect logits at each step
            max_len = self.max_answer_length
            curr_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device) # <s> token
            
            all_logits = []
            for _ in range(max_len):
                tgt = self.embedding(curr_tokens)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=device)
                out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
                next_token_logits = self.fc(out[:, -1:, :])
                all_logits.append(next_token_logits)
                next_token = next_token_logits.argmax(-1)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
                
            return torch.cat(all_logits, dim=1)
