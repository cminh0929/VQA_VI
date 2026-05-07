import torch
import torch.nn as nn
from .encoders import ImageEncoder, TextEncoder
from .fusion import FusionModule
from .decoders import LSTMDecoder, TransformerDecoder

class ModularVQA(nn.Module):
    def __init__(self, config, vocab_size, decoder_type='lstm'):
        super().__init__()
        self.decoder_type = decoder_type
        
        # 1. Encoders
        self.image_encoder = ImageEncoder(model_type=config.IMAGE_ENCODER, embed_size=config.EMBED_SIZE)
        self.text_encoder = TextEncoder(model_name=config.TEXT_ENCODER, embed_size=config.EMBED_SIZE)
        self.category_embedding = nn.Embedding(config.NUM_CATEGORIES, config.EMBED_SIZE)
        
        # 2. Fusion (Primarily for A1 Global context)
        self.fusion = FusionModule(embed_size=config.EMBED_SIZE)
        
        # 3. Decoders
        if decoder_type == 'lstm':
            self.decoder = LSTMDecoder(vocab_size, config.EMBED_SIZE, config.HIDDEN_SIZE, max_answer_length=config.MAX_ANSWER_LENGTH)
        else:
            self.decoder = TransformerDecoder(vocab_size, config.EMBED_SIZE, max_answer_length=config.MAX_ANSWER_LENGTH)

    def forward(self, images, input_ids, attention_mask, category_id, target_ids=None):
        # 1. Feature Extraction
        img_global, img_seq = self.image_encoder(images)
        txt_global, txt_seq = self.text_encoder(input_ids, attention_mask)
        cat_feat = self.category_embedding(category_id)
        
        if self.decoder_type == 'lstm':
            # A1: Use Global Fusion
            fused_global = self.fusion(img_global, txt_global, cat_feat)
            logits = self.decoder(fused_global, target_ids)
        else:
            # A2: Use Sequence Fusion / Cross-Attention
            # memory = [Image Patches + Text Tokens + Category]
            # cat_feat: [batch, embed_size] -> [batch, 1, embed_size]
            cat_seq = cat_feat.unsqueeze(1)
            # Concatenate all sequences into a single Memory bank
            memory = torch.cat([img_seq, txt_seq, cat_seq], dim=1) # [batch, 49+L+1, embed_size]
            logits = self.decoder(memory, target_ids)
            
        return logits
