import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, query_dim, key_dim):
        super(SimpleAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, key_dim)
        
    def forward(self, query, keys):
        # query: [batch, query_dim] -> [batch, 1, query_dim]
        # keys: [batch, num_keys, key_dim]
        query = self.query_proj(query).unsqueeze(1) # [batch, 1, key_dim]
        
        # dot product attention
        scores = torch.bmm(query, keys.transpose(1, 2)) # [batch, 1, num_keys]
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.bmm(attn_weights, keys) # [batch, 1, key_dim]
        return context.squeeze(1), attn_weights

class MultimodalFusion(nn.Module):
    def __init__(self, visual_dim, textual_dim, output_dim):
        super(MultimodalFusion, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.textual_proj = nn.Linear(textual_dim, output_dim)
        self.attention = SimpleAttention(output_dim, output_dim)
        
    def forward(self, visual_features, textual_features):
        # visual_features: [batch, num_patches, visual_dim]
        # textual_features: [batch, seq_len, textual_dim]
        
        v_proj = self.visual_proj(visual_features) # [batch, num_patches, output_dim]
        t_proj = self.textual_proj(textual_features) # [batch, seq_len, output_dim]
        
        # Use mean of text as query for visual attention
        query = t_proj.mean(dim=1)
        v_context, _ = self.attention(query, v_proj)
        
        # Concatenate attended visual and average textual
        fused = torch.cat([v_context, query], dim=-1)
        return fused # [batch, 2 * output_dim]
