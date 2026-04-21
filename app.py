import gradio as gr
import torch
from PIL import Image
from config import Config
# Import model loaders here

def predict(image, question, model_type):
    # This function will load the specified model (Direction A or B)
    # and perform inference
    if model_type == "Modular (Direction A)":
        return "Kết quả từ mô hình Modular"
    else:
        return "Kết quả từ mô hình PaliGemma (Hướng B)"

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
