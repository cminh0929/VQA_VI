import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.modular_vqa import ModularVQA
from utils.data_loader import get_dataloader
from config import Config
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.metrics import VQAMetrics

def train_modular(decoder_type='lstm', num_epochs=5, batch_size=16, learning_rate=1e-4):
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 0. Load Tokenizer (PhoBERT)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    metrics_calc = VQAMetrics(tokenizer)

    # 1. Load Data
    train_loader = get_dataloader(config, config.TRAIN_JSON, tokenizer=tokenizer, is_train=True, batch_size=batch_size)
    val_loader = get_dataloader(config, config.VAL_JSON, tokenizer=tokenizer, is_train=False, batch_size=batch_size)

    # 2. Initialize Model
    model = ModularVQA(config, decoder_type=decoder_type, vocab_size=len(tokenizer)).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            input_ids = batch['question'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['answer'].to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask, target_ids=target_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # 4. Validation Phase
        model.eval()
        val_stats = {'acc': [], 'bleu': [], 'rouge': []}
        print(f"\nEvaluating Epoch {epoch+1}...")
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(device)
                input_ids = batch['question'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_ids = batch['answer'].to(device)

                # Inference mode (no target_ids for teacher forcing)
                logits = model(images, input_ids, attention_mask)
                
                batch_m = metrics_calc.compute_batch_metrics(logits, target_ids)
                val_stats['acc'].append(batch_m['accuracy'])
                val_stats['bleu'].append(batch_m['bleu'])
                val_stats['rouge'].append(batch_m['rougeL'])

        print(f"Results - Acc: {np.mean(val_stats['acc']):.4f}, BLEU: {np.mean(val_stats['bleu']):.4f}, ROUGE-L: {np.mean(val_stats['rouge']):.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(config.CHECKPOINT_DIR, f"modular_{decoder_type}_epoch{epoch+1}.pt")
        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    # Train A1 (light version: 1 epoch)
    train_modular(decoder_type='lstm', num_epochs=1)
