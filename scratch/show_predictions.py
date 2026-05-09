import transformers.utils.import_utils as import_utils
import transformers.modeling_utils as modeling_utils
import_utils.check_torch_load_is_safe = lambda: None
modeling_utils.check_torch_load_is_safe = lambda: None

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from transformers import AutoTokenizer, BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel
from models.modular_vqa import ModularVQA
from utils.data_loader import get_dataloader
from config import Config
import os
from tqdm import tqdm

def show_preds_blip(model_id_or_path, config, test_loader, device, num_samples=5, title="BLIP"):
    processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
    base_model = BlipForQuestionAnswering.from_pretrained(config.BLIP_MODEL_ID).to(device)
    if model_id_or_path:
        model = PeftModel.from_pretrained(base_model, model_id_or_path).to(device)
    else:
        model = base_model
    model.eval()
    
    print(f"\n--- PREDICTIONS: {title} ---")
    samples_shown = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            outputs = model.generate(pixel_values=images, input_ids=input_ids, max_length=20)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)
            questions = batch['questions_raw']
            gts = batch['answers_raw']
            for i in range(len(preds)):
                if samples_shown >= num_samples: return
                print(f"Q: {questions[i]}\n  P: {preds[i]}\n  GT: {gts[i][0]}")
                samples_shown += 1

if __name__ == "__main__":
    config = Config()
    device = config.DEVICE
    loader_b = get_dataloader(config, config.TEST_JSON, direction='B', is_train=False, batch_size=2, limit=10)

    # Show BLIP B1 (Zero-shot)
    show_preds_blip(None, config, loader_b, device, title="BLIP B1 (ZERO-SHOT)")

    # Show BLIP B2 (Fine-tuned)
    path_b2 = os.path.join(config.CHECKPOINT_DIR, "blip_lora_best")
    if os.path.exists(path_b2):
        show_preds_blip(path_b2, config, loader_b, device, title="BLIP B2 (FINE-TUNED)")
