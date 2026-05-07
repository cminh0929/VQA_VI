import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # Fuse Global Image + Global Text + Category
        self.global_fusion = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, img_global, txt_global, cat_feat):
        """
        img_global: [batch, embed_size]
        txt_global: [batch, embed_size]
        cat_feat: [batch, embed_size]
        """
        combined = torch.cat([img_global, txt_global, cat_feat], dim=-1)
        return self.global_fusion(combined)
