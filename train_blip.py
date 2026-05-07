import torch
from torch.optim import AdamW
from tqdm import tqdm
import os
from config import Config
from models.blip_vqa import BlipVQAModel
from utils.data_loader import get_dataloader

def train_blip():
    config = Config()
    config.ensure_dirs()
    device = config.DEVICE
    
    model = BlipVQAModel.get_model(config, is_train=True)
    processor = BlipVQAModel.get_processor(config)
    
    train_loader = get_dataloader(config, config.TRAIN_JSON, direction='B', is_train=True)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE_B, weight_decay=config.WEIGHT_DECAY)
    
    print("Starting Training: Direction B - BLIP Fine-tuning (LoRA)")
    
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for step, batch in enumerate(pbar):
            # Only keep tensors for model input
            model_inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            optimizer.zero_grad()
            outputs = model(**model_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Save LoRA weights
        save_path = os.path.join(config.CHECKPOINT_DIR, f"blip_lora_epoch{epoch+1}")
        model.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_blip()
