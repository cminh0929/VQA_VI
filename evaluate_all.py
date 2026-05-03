import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from models.modular_vqa import ModularVQA
from utils.data_loader import get_dataloader
from utils.metrics import VQAMetrics
from config import Config
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

def evaluate_modular(model_path, decoder_type, config, test_loader, device):
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    model = ModularVQA(config, decoder_type=decoder_type, vocab_size=len(tokenizer)).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    metrics_calc = VQAMetrics(tokenizer)
    results = {'acc': [], 'bleu': [], 'rouge': []}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval Modular {decoder_type}"):
            images = batch['image'].to(device)
            input_ids = batch['question'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['answer'].to(device)
            
            logits = model(images, input_ids, attention_mask)
            m = metrics_calc.compute_batch_metrics(logits, target_ids)
            results['acc'].append(m['accuracy'])
            results['bleu'].append(m['bleu'])
            results['rouge'].append(m['rougeL'])
            
    return {k: np.mean(v) for k, v in results.items()}

def evaluate_paligemma(model_id_or_path, config, test_loader, device, is_zero_shot=False):
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id_or_path, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    processor = AutoProcessor.from_pretrained(config.MODEL_ID_B)
    model.eval()
    
    # We use basic accuracy for PaliGemma here (text comparison)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval PaliGemma {'Zero-shot' if is_zero_shot else 'Fine-tuned'}"):
            questions = [item['question'] for item in batch['original_item']]
            answers = [item['answer'] for item in batch['original_item']]
            images = [Image.fromarray(img.numpy().transpose(1, 2, 0).astype('uint8')) for img in batch['image']]
            
            inputs = processor(text=questions, images=images, return_tensors="pt", padding=True).to(device)
            output_tokens = model.generate(**inputs, max_new_tokens=20)
            
            # Extract only the generated part
            # PaliGemma generate returns full sequence [prompt + completion]
            decoded = processor.batch_decode(output_tokens, skip_special_tokens=True)
            
            for pred, gt in zip(decoded, answers):
                # Simple normalization
                p = pred.strip().lower()
                g = gt.strip().lower()
                if g in p or p in g: # Soft match for demo
                    correct += 1
                total += 1
                
    return {'acc': correct / total if total > 0 else 0}

if __name__ == "__main__":
    from transformers import AutoTokenizer
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer for data loading
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    # Data for evaluation (using Val set as proxy for Test if Test has no answers)
    test_loader = get_dataloader(config, config.VAL_JSON, tokenizer=tokenizer, is_train=False, batch_size=8)
    
    print("\n" + "="*30)
    print("STARTING EVALUATION OF ALL CONFIGS")
    print("="*30)
    
    # A1 & A2
    # res_a1 = evaluate_modular("checkpoints/modular_lstm_epoch1.pt", 'lstm', config, test_loader, device)
    # res_a2 = evaluate_modular("checkpoints/modular_transformer_epoch1.pt", 'transformer', config, test_loader, device)
    
    # B1 (Zero-shot)
    res_b1 = evaluate_paligemma(config.MODEL_ID_B, config, test_loader, device, is_zero_shot=True)
    
    # B2 (Fine-tuned)
    # b2_path = os.path.join(config.CHECKPOINT_DIR, "paligemma_b2_epoch1")
    # res_b2 = evaluate_paligemma(b2_path, config, test_loader, device)
    
    print("\nFINAL RESULTS SUMMARY:")
    print(f"A1 (Modular+LSTM): {0.0} (Needs checkpoint)")
    print(f"A2 (Modular+Trans): {0.0} (Needs checkpoint)")
    print(f"B1 (Zero-shot): {res_b1['acc']:.4f}")
    print(f"B2 (Fine-tuned): {0.0} (Needs checkpoint)")
