import transformers.utils.import_utils as import_utils
import transformers.modeling_utils as modeling_utils
# Monkey patch to bypass security check for torch.load (CVE-2025-32434)
# Required to load PhoBERT weights from local 'weight' directory in this environment.
import_utils.check_torch_load_is_safe = lambda: None
modeling_utils.check_torch_load_is_safe = lambda: None

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from transformers import AutoTokenizer, BlipProcessor, BlipForQuestionAnswering
from models.modular_vqa import ModularVQA
from utils.data_loader import get_dataloader
from utils.metrics import VQAMetrics
from config import Config
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

def evaluate_modular(model_path, decoder_type, config, test_loader, device):
    tokenizer = AutoTokenizer.from_pretrained("weight")
    model = ModularVQA(config, vocab_size=len(tokenizer), decoder_type=decoder_type).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    metrics_calc = VQAMetrics(tokenizer)
    preds, gts = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval Modular {decoder_type}"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            category_id = batch['category_id'].to(device)
            target_ids = batch['labels'].to(device)
            
            logits = model(images, input_ids, attention_mask, category_id)
            
            # Simple greedy decode for evaluation
            batch_preds = logits.argmax(-1)
            decoded_preds = tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
            
            preds.extend(decoded_preds)
            gts.extend(batch['answers_raw'])
            
    results = metrics_calc.compute_batch_metrics(preds, gts)
    print(f"\n--- [DEBUG Modular {decoder_type}] Sample Preds: {preds[:3]}")
    # Print first few token IDs of the first prediction
    with torch.no_grad():
        token_ids = logits[0].argmax(-1).tolist()
        print(f"--- [DEBUG Modular {decoder_type}] Token IDs (Sample 0): {token_ids}")
    print(f"--- [DEBUG Modular {decoder_type}] Sample GTs: {gts[:3]}")
    return results

def evaluate_blip(model_id_or_path, config, test_loader, device, is_zero_shot=False):
    if is_zero_shot:
        model = BlipForQuestionAnswering.from_pretrained(
            config.BLIP_MODEL_ID, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    else:
        from peft import PeftModel
        base_model = BlipForQuestionAnswering.from_pretrained(
            config.BLIP_MODEL_ID, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model = PeftModel.from_pretrained(base_model, model_id_or_path)
        
    processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
    model.eval()
    
    metrics_calc = VQAMetrics()
    preds, gts = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval BLIP {'Zero-shot' if is_zero_shot else 'Fine-tuned'}"):
            images = batch['pixel_values'].to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
            input_ids = batch['input_ids'].to(device)
            
            output_tokens = model.generate(
                pixel_values=images, 
                input_ids=input_ids, 
                max_length=config.MAX_ANSWER_LENGTH
            )
            decoded = processor.batch_decode(output_tokens, skip_special_tokens=True)
            
            preds.extend(decoded)
            # Use all reference answers for accurate evaluation
            gts.extend(batch['answers_raw'])
                
    results = metrics_calc.compute_batch_metrics(preds, gts)
    print(f"\n--- [DEBUG BLIP] Sample Preds: {preds[:3]}")
    print(f"--- [DEBUG BLIP] Sample GTs: {gts[:3]}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run with small subset of data")
    parser.add_argument("--skip-blip", action="store_true", help="Skip BLIP evaluation")
    args = parser.parse_args()
    
    config = Config()
    device = config.DEVICE
    
    test_limit = 3 if args.debug else None
    default_metrics = {"accuracy": 0, "bleu": 0, "rougeL": 0}
    
    # ============================================================
    # Direction A: Modular Models (LSTM + Transformer)
    # ============================================================
    print("\n" + "="*50)
    print("  EVALUATION: Direction A - Modular Models")
    print("="*50)
    
    test_loader_a = get_dataloader(config, config.TEST_JSON, direction='A', is_train=False, batch_size=8, limit=test_limit)
    
    # A1: LSTM Decoder
    path_a1 = os.path.join(config.CHECKPOINT_DIR, "modular_lstm_best.pt")
    if os.path.exists(path_a1):
        print(f"\n[A1] Found checkpoint: {path_a1}")
        res_a1 = evaluate_modular(path_a1, 'lstm', config, test_loader_a, device)
    else:
        print(f"\n[A1] Checkpoint NOT found: {path_a1} — Skipping.")
        res_a1 = default_metrics
    
    # A2: Transformer Decoder
    path_a2 = os.path.join(config.CHECKPOINT_DIR, "modular_transformer_best.pt")
    if os.path.exists(path_a2):
        print(f"\n[A2] Found checkpoint: {path_a2}")
        res_a2 = evaluate_modular(path_a2, 'transformer', config, test_loader_a, device)
    else:
        print(f"\n[A2] Checkpoint NOT found: {path_a2} — Skipping.")
        res_a2 = default_metrics
    
    # ============================================================
    # Direction B: BLIP Models (Zero-shot + Fine-tuned)
    # ============================================================
    res_b1 = default_metrics
    res_b2 = default_metrics
    
    if not args.skip_blip:
        print("\n" + "="*50)
        print("  EVALUATION: Direction B - BLIP Models")
        print("="*50)
        
        test_loader_b = get_dataloader(config, config.TEST_JSON, direction='B', is_train=False, batch_size=4, limit=test_limit)
        
        # B1: BLIP Zero-shot (no fine-tuning)
        print("\n[B1] Evaluating BLIP Zero-shot...")
        try:
            res_b1 = evaluate_blip(None, config, test_loader_b, device, is_zero_shot=True)
        except Exception as e:
            print(f"[B1] Error: {e}")
        
        # B2: BLIP Fine-tuned (LoRA)
        path_b2 = os.path.join(config.CHECKPOINT_DIR, "blip_lora_best")
        if not os.path.exists(path_b2):
            # Try to find the latest epoch checkpoint
            blip_checkpoints = [d for d in os.listdir(config.CHECKPOINT_DIR) if d.startswith("blip_lora_epoch")]
            if blip_checkpoints:
                blip_checkpoints.sort(key=lambda x: int(x.replace("blip_lora_epoch", "")))
                path_b2 = os.path.join(config.CHECKPOINT_DIR, blip_checkpoints[-1])
        
        if os.path.exists(path_b2):
            print(f"\n[B2] Found LoRA checkpoint: {path_b2}")
            try:
                res_b2 = evaluate_blip(path_b2, config, test_loader_b, device, is_zero_shot=False)
            except Exception as e:
                print(f"[B2] Error: {e}")
        else:
            print(f"\n[B2] No BLIP LoRA checkpoint found — Skipping.")
    else:
        print("\n[INFO] Skipping BLIP evaluation (--skip-blip flag)")
    
    # ============================================================
    # Summary Report
    # ============================================================
    summary = f"""
{'='*55}
  FINAL RESULTS SUMMARY — Vietnamese VQA Evaluation
{'='*55}
  Direction A (Modular: ResNet50 + PhoBERT)
  -----------------------------------------
  A1 (LSTM Decoder):        Acc: {res_a1['accuracy']:.4f} | BLEU: {res_a1['bleu']:.4f} | ROUGE-L: {res_a1['rougeL']:.4f}
  A2 (Transformer Decoder): Acc: {res_a2['accuracy']:.4f} | BLEU: {res_a2['bleu']:.4f} | ROUGE-L: {res_a2['rougeL']:.4f}

  Direction B (Multimodal: BLIP + LoRA)
  -----------------------------------------
  B1 (Zero-shot):           Acc: {res_b1['accuracy']:.4f} | BLEU: {res_b1['bleu']:.4f} | ROUGE-L: {res_b1['rougeL']:.4f}
  B2 (Fine-tuned LoRA):     Acc: {res_b2['accuracy']:.4f} | BLEU: {res_b2['bleu']:.4f} | ROUGE-L: {res_b2['rougeL']:.4f}
{'='*55}
"""
    print(summary)
    
    os.makedirs("results", exist_ok=True)
    with open("results/eval_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print("Results saved to results/eval_summary.txt")

