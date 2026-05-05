import gradio as gr
import torch
import os
from dotenv import load_dotenv
load_dotenv()
import glob
from PIL import Image
from config import Config
from transformers import modeling_utils, masking_utils
modeling_utils.check_torch_load_is_safe = lambda: None
masking_utils._is_torch_greater_or_equal_than_2_6 = True
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, AutoProcessor
from models.modular_vqa import ModularVQA
from peft import PeftModel
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Caches ---
_modular_tokenizer = None
_paligemma_processor = None
_loaded_models = {}

_transform = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def _find_latest_checkpoint(prefix):
    # Find the latest epoch checkpoint matching prefix in CHECKPOINT_DIR
    files = glob.glob(os.path.join(config.CHECKPOINT_DIR, f"{prefix}*"))
    if not files:
        return None
    # Sort to try to get the latest epoch if possible
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def _get_modular_model(decoder_type):
    global _modular_tokenizer
    model_key = f"modular_{decoder_type}"
    
    if model_key in _loaded_models:
        return _loaded_models[model_key], _modular_tokenizer
        
    if _modular_tokenizer is None:
        _modular_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        
    model = ModularVQA(config, decoder_type=decoder_type, vocab_size=len(_modular_tokenizer)).to(device)
    
    ckpt = _find_latest_checkpoint(f"modular_{decoder_type}_epoch")
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Successfully loaded {model_key} checkpoint: {ckpt}")
    else:
        print(f"Warning: No checkpoint found for {model_key}. Using uninitialized weights.")
        
    model.eval()
    _loaded_models[model_key] = model
    return model, _modular_tokenizer

def _get_paligemma_model(is_finetuned):
    global _paligemma_processor
    model_key = "paligemma_finetuned" if is_finetuned else "paligemma_zeroshot"
    
    if model_key in _loaded_models:
        return _loaded_models[model_key], _paligemma_processor
        
    if _paligemma_processor is None:
        _paligemma_processor = AutoProcessor.from_pretrained(config.MODEL_ID_B)
        
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.MODEL_ID_B,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)
    
    if is_finetuned:
        ckpt = _find_latest_checkpoint("paligemma_b2_epoch")
        if ckpt and os.path.exists(ckpt):
            model = PeftModel.from_pretrained(base_model, ckpt)
            print(f"Successfully loaded {model_key} LoRA checkpoint: {ckpt}")
        else:
            print(f"Warning: No LoRA checkpoint found. Falling back to base model.")
            model = base_model
    else:
        model = base_model
        
    model.eval()
    _loaded_models[model_key] = model
    return model, _paligemma_processor

def predict(image, question, model_choice):
    if image is None or not question.strip():
        return "⚠️ Vui lòng tải ảnh và nhập câu hỏi."
    
    try:
        if model_choice.startswith("A1"):
            model, tokenizer = _get_modular_model('lstm')
            is_modular = True
        elif model_choice.startswith("A2"):
            model, tokenizer = _get_modular_model('transformer')
            is_modular = True
        elif model_choice.startswith("B1"):
            model, processor = _get_paligemma_model(is_finetuned=False)
            is_modular = False
        elif model_choice.startswith("B2"):
            model, processor = _get_paligemma_model(is_finetuned=True)
            is_modular = False
            
        if is_modular:
            img_arr = np.array(image.convert('RGB'))
            img_tensor = _transform(image=img_arr)['image'].unsqueeze(0).to(device)
            inputs = tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attn_mask = inputs['attention_mask'].to(device)
            
            with torch.no_grad():
                logits = model(img_tensor, input_ids, attn_mask)
            token_ids = logits.argmax(-1)[0].tolist()
            return tokenizer.decode(token_ids, skip_special_tokens=True)
            
        else:
            inputs = processor(text=question, images=image, return_tensors='pt').to(device)
            input_len = inputs['input_ids'].shape[1]
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20)
            generated_tokens = out[:, input_len:]
            return processor.decode(generated_tokens[0], skip_special_tokens=True)
            
    except Exception as e:
        return f"Lỗi trong quá trình dự đoán: {str(e)}"

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🇻🇳 Vietnamese Visual Question Answering")
        gr.Markdown("Hệ thống giải đáp thắc mắc qua hình ảnh bằng tiếng Việt (Hỗ trợ 4 Mô hình).")
        
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Tải ảnh lên")
                question_input = gr.Textbox(label="Câu hỏi", placeholder="Ví dụ: Đây là đâu?")
                model_selector = gr.Radio(
                    [
                        "A1: Modular VQA (LSTM Decoder)", 
                        "A2: Modular VQA (Transformer Decoder)",
                        "B1: PaliGemma (Zero-shot)",
                        "B2: PaliGemma (Fine-tuned / LoRA)"
                    ], 
                    label="Chọn Mô hình", 
                    value="B2: PaliGemma (Fine-tuned / LoRA)"
                )
                submit_btn = gr.Button("Phân tích", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(label="Câu trả lời")
        
        submit_btn.click(
            fn=predict,
            inputs=[img_input, question_input, model_selector],
            outputs=output_text
        )
        
        gr.Markdown("### Ảnh tham khảo (Click để test ngay)")
        gr.Examples(
            examples=[
                [os.path.join(config.IMAGES_DIR, "antelope_1.jpg"), "Đây là con gì?", "B2: PaliGemma (Fine-tuned / LoRA)"],
                [os.path.join(config.IMAGES_DIR, "antelope_10.jpg"), "Con vật trong ảnh là gì?", "B2: PaliGemma (Fine-tuned / LoRA)"]
            ],
            inputs=[img_input, question_input, model_selector]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(theme=gr.themes.Soft())
