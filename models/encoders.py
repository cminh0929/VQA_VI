import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoConfig, AutoTokenizer

class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(ImageEncoder, self).__init__()
        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            self.out_channels = 2048
        elif model_name == 'vit':
            # Future expansion for ViT
            pass
        
    def forward(self, x):
        # x: [batch, 3, 224, 224]
        features = self.feature_extractor(x) # [batch, 2048, 7, 7]
        features = features.permute(0, 2, 3, 1) # [batch, 7, 7, 2048]
        features = features.view(features.size(0), -1, self.out_channels) # [batch, 49, 2048]
        return features

class TextEncoder(nn.Module):
    def __init__(self, model_name='vinai/phobert-base', freeze=True):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Using the last hidden state
        return outputs.last_hidden_state # [batch, seq_len, 768]
