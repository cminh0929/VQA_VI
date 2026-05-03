import gradio as gr
import torch
import os
from PIL import Image
from config import Config
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, AutoProcessor
from models.modular_vqa import ModularVQA

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(image, question, model_type):
    if image is None or not question.strip():
        return "⚠️ Vui lòng tải ảnh và nhập câu hỏi."
    
    if model_type == "Modular (Direction A)":
        try:
            tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
            model = ModularVQA(config, decoder_type='lstm', vocab_size=len(tokenizer)).to(device)
            ckpt = os.path.join(config.CHECKPOINT_DIR, "modular_lstm_epoch1.pt")
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device))
            else:
                return "⚠️ Chưa có checkpoint Hướng A. Vui lòng train trước."
            model.eval()
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            import numpy as np
            import cv2
            transform = A.Compose([
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            img_arr = np.array(image.convert('RGB'))
            img_tensor = transform(image=img_arr)['image'].unsqueeze(0).to(device)
            inputs = tokenizer(question, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attn_mask = inputs['attention_mask'].to(device)
            with torch.no_grad():
                logits = model(img_tensor, input_ids, attn_mask)
            token_ids = logits.argmax(-1)[0].tolist()
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            return f"Lỗi: {str(e)}"
    else:
        try:
            processor = AutoProcessor.from_pretrained(config.MODEL_ID_B)
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                config.MODEL_ID_B,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            ).to(device)
            model.eval()
            inputs = processor(text=question, images=image, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20)
            return processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return f"Lỗi (PaliGemma chưa được tải hoặc không đủ bộ nhớ): {str(e)}"

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🇻🇳 Vietnamese Visual Question Answering")
        gr.Markdown("Hệ thống giải đáp thắc mắc qua hình ảnh bằng tiếng Việt.")
        
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Tải ảnh lên")
                question_input = gr.Textbox(label="Câu hỏi", placeholder="Ví dụ: Đây là đâu?")
                model_selector = gr.Radio(
                    ["Modular (Direction A)", "PaliGemma (Direction B)"], 
                    label="Chọn Mô hình", 
                    value="PaliGemma (Direction B)"
                )
                submit_btn = gr.Button("Phân tích", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(label="Câu trả lời")
        
        submit_btn.click(
            fn=predict,
            inputs=[img_input, question_input, model_selector],
            outputs=output_text
        )
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
