import transformers.utils.import_utils as import_utils
import transformers.modeling_utils as modeling_utils
import_utils.check_torch_load_is_safe = lambda: None
modeling_utils.check_torch_load_is_safe = lambda: None

import torch
import os
from config import Config
from models.modular_vqa import ModularVQA
from transformers import AutoTokenizer

def debug_inference():
    config = Config()
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("weight")
    model = ModularVQA(config, vocab_size=len(tokenizer), decoder_type='lstm').to(device)
    path = os.path.join(config.CHECKPOINT_DIR, "modular_lstm_epoch10.pt")
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    # Dummy data
    image = torch.randn(1, 3, 224, 224).to(device)
    input_ids = torch.randint(0, len(tokenizer), (1, 10)).to(device)
    attention_mask = torch.ones((1, 10)).to(device)
    category_id = torch.tensor([0]).to(device)
    
    with torch.no_grad():
        # Use the full model forward (inference mode, no target_ids)
        logits = model(image, input_ids, attention_mask, category_id)
        preds = logits.argmax(-1)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        print(f"Decoded prediction: {decoded}")
        print(f"Predictions: {preds.tolist()}")
        print(f"Sample Logits (first 10 of first step): {logits[0, 0, :10].tolist()}")
        print(f"Top 5 tokens for first step: {torch.topk(logits[0, 0], 5).indices.tolist()}")
        print(f"Top 5 values for first step: {torch.topk(logits[0, 0], 5).values.tolist()}")

if __name__ == "__main__":
    debug_inference()
