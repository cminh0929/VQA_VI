import gradio as gr
import torch
from PIL import Image
from config import Config
from models.modular_vqa import ModularVQA
from models.blip_vqa import BlipVQAModel
from transformers import AutoTokenizer, BlipForConditionalGeneration, BlipProcessor
from peft import PeftModel
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from underthesea import word_tokenize

config = Config()
device = config.DEVICE

# --- Cache ---
_models = {}
_tokenizers = {}
_processors = {}

def get_modular_model(decoder_type):
    model_key = f"modular_{decoder_type}"
    if model_key not in _models:
        tokenizer = AutoTokenizer.from_pretrained("weight")
        model = ModularVQA(config, vocab_size=len(tokenizer), decoder_type=decoder_type).to(device)
        path = os.path.join(config.CHECKPOINT_DIR, f"modular_{decoder_type}_epoch10.pt")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        _models[model_key] = model
        _tokenizers[model_key] = tokenizer
    return _models[model_key], _tokenizers[model_key]

def get_blip_model(is_finetuned=True):
    model_key = "blip_ft" if is_finetuned else "blip_zs"
    if model_key not in _models:
        processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_ID)
        if is_finetuned:
            base = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL_ID).to(device)
            path = os.path.join(config.CHECKPOINT_DIR, "blip_lora_epoch10")
            if os.path.exists(path):
                model = PeftModel.from_pretrained(base, path)
            else:
                model = base
        else:
            model = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL_ID).to(device)
        model.eval()
        _models[model_key] = model
        _processors[model_key] = processor
    return _models[model_key], _processors[model_key]

def predict(image, question, mode):
    if image is None or not question:
        return "Vui lòng cung cấp ảnh và câu hỏi."
    
    question_seg = word_tokenize(question.lower(), format="text")
    
    try:
        if mode.startswith("A"):
            decoder = 'lstm' if "LSTM" in mode else 'transformer'
            model, tokenizer = get_modular_model(decoder)
            
            # Preprocess image
            transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
            img_tensor = transform(image=np.array(image))['image'].unsqueeze(0).to(device)
            
            # Preprocess text
            tokens = tokenizer(question_seg, return_tensors="pt", padding=True).to(device)
            
            # Default category_id = 0 (object) when not provided by UI
            category_id = torch.tensor([0], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(img_tensor, tokens['input_ids'], tokens['attention_mask'], category_id)
                preds = logits.argmax(-1)
                return tokenizer.decode(preds[0], skip_special_tokens=True)
                
        else:
            is_ft = "Fine-tuned" in mode
            model, processor = get_blip_model(is_ft)
            inputs = processor(images=image, text=question_seg, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10)
                return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Lỗi: {str(e)}"

def main():
    with gr.Blocks(title="Vietnamese VQA Demo") as demo:
        gr.Markdown("# 🇻🇳 Vietnamese Visual Question Answering")
        gr.Markdown("Hệ thống hỏi đáp hình ảnh tiếng Việt chuyên biệt. Chọn cấu hình và đặt câu hỏi.")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Ảnh đầu vào")
                input_txt = gr.Textbox(label="Câu hỏi tiếng Việt", placeholder="Ví dụ: Con mèo màu gì?")
                mode_select = gr.Radio(
                    ["A1: Modular + LSTM Decoder", "A2: Modular + Transformer Decoder", 
                     "B1: BLIP Zero-shot", "B2: BLIP Fine-tuned"],
                    label="Cấu hình mô hình",
                    value="B2: BLIP Fine-tuned"
                )
                btn = gr.Button("Trả lời", variant="primary")
            with gr.Column():
                output_txt = gr.Textbox(label="Kết quả dự đoán")
        
        btn.click(predict, inputs=[input_img, input_txt, mode_select], outputs=output_txt)
        
    demo.launch()

if __name__ == "__main__":
    main()
