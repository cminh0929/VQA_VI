import torch
from transformers import modeling_utils, masking_utils
modeling_utils.check_torch_load_is_safe = lambda: None
masking_utils._is_torch_greater_or_equal_than_2_6 = True
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor, AutoTokenizer
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
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.MODEL_ID_B, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    if is_zero_shot:
        model = base_model
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_id_or_path)
        
    processor = AutoProcessor.from_pretrained(config.MODEL_ID_B)
    model.eval()
    
    # We use basic accuracy for PaliGemma here (text comparison)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval PaliGemma {'Zero-shot' if is_zero_shot else 'Fine-tuned'}"):
            questions = batch['question']
            answers = batch['answer']
            images = [Image.fromarray(img.numpy().transpose(1, 2, 0).astype('uint8')) for img in batch['image']]
            
            inputs = processor(text=questions, images=images, return_tensors="pt", padding=True).to(device)
            input_len = inputs['input_ids'].shape[1]
            output_tokens = model.generate(**inputs, max_new_tokens=20)
            
            # Extract only the generated part (strip prompt tokens)
            generated_tokens = output_tokens[:, input_len:]
            decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for pred, gt in zip(decoded, answers):
                # Simple normalization
                p = pred.strip().lower()
                g = gt.strip().lower()
                if p == g:  # Exact match
                    correct += 1
                total += 1
                
    return {'acc': correct / total if total > 0 else 0}

if __name__ == "__main__":
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer for data loading
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    # Data for evaluation (using Test set)
    test_loader = get_dataloader(config, config.TEST_JSON, tokenizer=tokenizer, is_train=False, batch_size=8)
    # Separate loader for PaliGemma (no normalization)
    test_loader_pali = get_dataloader(config, config.TEST_JSON, tokenizer=None, is_train=False, batch_size=4, normalize=False)
    
    print("\n" + "="*30)
    print("STARTING EVALUATION OF ALL CONFIGS")
    print("="*30)
    
    # A1 & A2
    res_a1 = evaluate_modular("results/checkpoints/modular_lstm_epoch9.pt", 'lstm', config, test_loader, device)
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    res_a2 = evaluate_modular("results/checkpoints/modular_transformer_epoch9.pt", 'transformer', config, test_loader, device)
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    # B1 (Zero-shot)
    res_b1 = evaluate_paligemma(config.MODEL_ID_B, config, test_loader_pali, device, is_zero_shot=True)
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    # B2 (Fine-tuned)
    b2_path = os.path.join(config.CHECKPOINT_DIR, "paligemma_b2_epoch_final")
    res_b2 = evaluate_paligemma(b2_path, config, test_loader_pali, device, is_zero_shot=False)
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    summary = f"""
FINAL RESULTS SUMMARY:
A1 (Modular+LSTM): Acc: {res_a1['acc']:.4f} | BLEU: {res_a1['bleu']:.4f} | ROUGE-L: {res_a1['rouge']:.4f}
A2 (Modular+Trans): Acc: {res_a2['acc']:.4f} | BLEU: {res_a2['bleu']:.4f} | ROUGE-L: {res_a2['rouge']:.4f}
B1 (Zero-shot): Acc: {res_b1['acc']:.4f}
B2 (Fine-tuned): Acc: {res_b2['acc']:.4f}
"""
    print(summary)
    with open("results/eval_results.txt", "w", encoding="utf-8") as f:
        f.write(summary)
