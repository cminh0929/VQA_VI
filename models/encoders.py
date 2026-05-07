import torch
import torch.nn as nn
import os
from torchvision import models
from transformers import AutoModel, AutoConfig

class ImageEncoder(nn.Module):
    def __init__(self, model_type="resnet50", embed_size=768):
        super().__init__()
        if model_type == "resnet50":
            # Load local weights
            base = models.resnet50(weights=None)
            local_weights = "weight/resnet50-0676ba61.pth"
            if os.path.exists(local_weights):
                print(f"Loading local ResNet50 weights from {local_weights}")
                base.load_state_dict(torch.load(local_weights, weights_only=False))
            else:
                print("Local ResNet50 weights not found, attempting download...")
                base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.avgpool = base.avgpool
            self.projection = nn.Linear(base.fc.in_features, embed_size)
            self.spatial_projection = nn.Conv2d(base.fc.in_features, embed_size, kernel_size=1)
        else:
            raise ValueError("Only resnet50 is standardized for now")

    def forward(self, x):
        spatial_features = self.backbone(x)
        global_features = self.avgpool(spatial_features)
        global_features = torch.flatten(global_features, 1)
        global_features = self.projection(global_features)
        spatial_features = self.spatial_projection(spatial_features)
        spatial_features = spatial_features.flatten(2).transpose(1, 2)
        return global_features, spatial_features


class TextEncoder(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", embed_size=768):
        super().__init__()
        local_bin = "weight/pytorch_model.bin"
        local_config = "weight/config.json"
        
        if os.path.exists(local_bin) and os.path.exists(local_config):
            print(f"Loading PhoBERT in 100% OFFLINE mode from 'weight' folder...")
            try:
                self.bert = AutoModel.from_pretrained("weight", local_files_only=True)
            except Exception as e:
                print(f"CRITICAL: Failed to load PhoBERT offline: {e}")
                raise e
        else:
            print("PhoBERT local files not found, attempting online load (will likely fail if no internet/security issues)")
            self.bert = AutoModel.from_pretrained(model_name)
            
        self.projection = nn.Linear(self.bert.config.hidden_size, embed_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Sequence output: [batch, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        sequence_output = self.projection(sequence_output)
        
        # Pooled output ([CLS] token): [batch, hidden_size]
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = sequence_output[:, 0, :]
        else:
            pooled_output = self.projection(pooled_output)
            
        return pooled_output, sequence_output
