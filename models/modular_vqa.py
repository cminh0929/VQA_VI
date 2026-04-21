import torch
import torch.nn as nn
from .encoders import ImageEncoder, TextEncoder
from .fusion import MultimodalFusion
from .decoders import LSTMDecoder, TransformerDecoder

class ModularVQA(nn.Module):
    def __init__(self, config, decoder_type='transformer', vocab_size=64000):
        super(ModularVQA, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        self.fusion = MultimodalFusion(
            visual_dim=2048, 
            textual_dim=768, 
            output_dim=512
        )
        
        if decoder_type == 'lstm':
            self.decoder = LSTMDecoder(512, 512, vocab_size)
        else:
            self.decoder = TransformerDecoder(512, 512, vocab_size)
            
    def forward(self, images, input_ids, attention_mask, target_ids=None):
        v_features = self.image_encoder(images)
        t_features = self.text_encoder(input_ids, attention_mask)
        
        fused = self.fusion(v_features, t_features)
        
        logits = self.decoder(fused, target_ids)
        return logits
