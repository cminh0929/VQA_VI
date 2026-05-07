import sys
import os
import matplotlib.pyplot as plt
import torch
from transformers import BlipProcessor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.data_loader import get_dataloader

def verify_pipeline():
    config = Config()
    print("--- Initializing BLIP Processor ---")
    processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
    
    print(f"--- Loading Dataset from {config.TRAIN_JSON} ---")
    if not os.path.exists(config.TRAIN_JSON):
        print(f"Error: {config.TRAIN_JSON} not found. Please check your data path.")
        return

    # Initialize DataLoader
    train_loader = get_dataloader(config, config.TRAIN_JSON, processor, batch_size=4, is_train=True)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print("\nBatch Information:")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Pixel Values shape: {batch['pixel_values'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # Decode some samples
    questions = processor.batch_decode(batch['input_ids'], skip_special_tokens=True)
    answers = processor.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    
    print("\nSample Data (First 2 items):")
    for i in range(min(2, len(questions))):
        print(f"Item {i}:")
        print(f"  Q: {questions[i]}")
        print(f"  A: {answers[i]}")
        print(f"  Image ID: {batch['image_ids'][i]}")

    # Visualize the first batch
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    pixel_values = batch['pixel_values'].cpu().numpy().transpose(0, 2, 3, 1)
    
    # Denormalize for visualization if needed, but BLIP normalization is standard
    # Standard mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(4):
        img = pixel_values[i]
        img = img * std + mean
        img = (img * 255).clip(0, 255).astype('uint8')
        
        axes[i].imshow(img)
        axes[i].set_title(f"Q: {questions[i][:20]}...\nA: {answers[i]}")
        axes[i].axis('off')
    
    output_viz = os.path.join(config.BASE_DIR, "results", "data_verification.png")
    os.makedirs(os.path.dirname(output_viz), exist_ok=True)
    plt.savefig(output_viz)
    print(f"\nVerification image saved to: {output_viz}")

if __name__ == "__main__":
    verify_pipeline()
