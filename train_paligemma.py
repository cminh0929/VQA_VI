import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from utils.data_loader import get_dataloader
from config import Config
from tqdm import tqdm
import os

def train_paligemma():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.MODEL_ID_B, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    processor = AutoProcessor.from_pretrained(config.MODEL_ID_B)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    
    # DataLoader - Note: for PaliGemma, we pass raw images/text to processor later
    # normalize=False because PaliGemma's processor handles its own normalization
    train_loader = get_dataloader(config, config.TRAIN_JSON, tokenizer=None, is_train=True, batch_size=config.BATCH_SIZE_B, normalize=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_B)
    
    # Ensure checkpoint directory exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Training Loop
    for epoch in range(config.EPOCHS_B):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS_B}")
        
        for batch in pbar:
            # Batch items from data_loader
            questions = [item['question'] for item in batch['original_item']]
            answers = [item['answer'] for item in batch['original_item']]
            images = [Image.fromarray(img.numpy().transpose(1, 2, 0).astype('uint8')) for img in batch['image']] # Need PIL format
            
            # Prepare inputs for PaliGemma
            inputs = processor(text=questions, images=images, suffix=answers, return_tensors="pt", padding=True).to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
        
        # Save B2 Checkpoint
        save_path = os.path.join(config.CHECKPOINT_DIR, f"paligemma_b2_epoch{epoch+1}")
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_paligemma()
