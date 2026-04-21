import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from utils.data_loader import get_dataloader
from config import Config

def train():
    config = Config()
    
    # Load model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained(config.MODEL_ID_B, torch_dtype=torch.float16)
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
    model.print_trainable_parameters()
    
    # DataLoader
    train_loader = get_dataloader(config, config.TRAIN_JSON, tokenizer=None, is_train=True, batch_size=config.BATCH_SIZE_B)
    
    # Training Loop (Simplified)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_B)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    model.train()
    for epoch in range(config.EPOCHS_B):
        for batch in train_loader:
            images = batch['image'].to(device)
            # PaliGemma processing logic
            print(f"Training on batch...")
            break # Just a placeholder
            
if __name__ == "__main__":
    train()
