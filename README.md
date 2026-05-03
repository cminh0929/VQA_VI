# DỰ ÁN CUỐI KỲ MÔN HỌC SÂU: Hệ thống Hỏi đáp trên Ảnh (Visual Question Answering)

Dự án xây dựng hệ thống Visual Question Answering (VQA) tiếng Việt trên một miền chuyên biệt. Hệ thống nhận đầu vào là ảnh và câu hỏi tiếng Việt, từ đó sinh ra câu trả lời tương ứng. Dự án kết hợp các kiến thức về mạng nơ-ron tích chập (CNN), mạng học sâu chuỗi (LSTM, Transformer) và học đa phương thức (Multimodal Learning).

## 1. Dữ liệu
- **Miền chuyên biệt**: OpenViVQA (Dữ liệu đa miền tại Việt Nam: thắng cảnh, ẩm thực, đời sống...)
- **Quy mô dữ liệu**:
  - Tập huấn luyện (Train): $\ge$ 2000 bộ (ảnh, câu hỏi, câu trả lời) với tối thiểu 200 ảnh và mỗi ảnh có $\ge$ 3 câu hỏi.
  - Tập kiểm thử (Test): $\ge$ 50 bộ chuẩn bị thủ công, ảnh không trùng lặp với tập train.
- **Loại câu hỏi**: Đa dạng (Yes/No, đếm số lượng, nhận dạng, thuộc tính, không gian...).
- **Đầu ra**: Câu trả lời ngắn ngọn (dưới 10 từ).
- **Phân chia dữ liệu**: Train / Val / Test theo tỷ lệ 80/10/10.
- **Tăng cường dữ liệu (Data Augmentation)**: Áp dụng các kỹ thuật tăng cường ảnh (lật, xoay, crop) và văn bản (paraphrase, back-translation).

## 2. Mô hình (Hai hướng tiếp cận)

### Hướng A — Kiến trúc rời
- **Image encoder**: Sử dụng CNN pretrained (ResNet/VGG/EfficientNet) hoặc ViT.
- **Text encoder**: Sử dụng LSTM/BiLSTM hoặc PhoBERT.
- **Fusion**: Thực hiện kết hợp đặc trưng qua concat, element-wise, hoặc co-attention.
- **Answer decoder**: So sánh giữa LSTM decoder và Transformer decoder (giữ nguyên encoder).

### Hướng B — Multimodal pretrained
- Thực hiện Fine-tune các mô hình: BLIP/BLIP-2, ViLT, LLaVA, Qwen-VL, hoặc PaliGemma (sử dụng LoRA/PEFT nếu cần).
- Chiến lược xử lý tiếng Việt: Dịch thuật trước khi đưa vào mô hình hoặc sử dụng mô hình hỗ trợ trực tiếp tiếng Việt.

## 3. Thực nghiệm và Đánh giá

### Cấu hình thực nghiệm
1. **A1**: Hướng A với LSTM decoder.
2. **A2**: Hướng A với Transformer decoder (So sánh với A1 để đánh giá ảnh hưởng của decoder).
3. **B1**: Hướng B ở chế độ zero-shot.
4. **B2**: Hướng B sau khi fine-tuned.

### Tiêu chí đánh giá
- VQA Accuracy (exact match / soft accuracy chuẩn VQA v2).
- Các độ đo xử lý ngôn ngữ: BLEU, ROUGE-L, METEOR.
- Đánh giá ngữ nghĩa: BERTScore.
- Phương pháp đánh giá LLM-as-a-judge.

## 4. Các giải pháp nâng cao chất lượng (Nâng cao)
- **Reinforcement Learning (RL)**: Huấn luyện bổ sung bằng RL (PPO với reward là VQA Accuracy/BERTScore, DPO, hoặc RLHF).
  - Yêu cầu Preference data $\ge$ 100 cặp.
  - So sánh kết quả RL và SFT (Supervised Fine-Tuning).
- Các kỹ thuật tối ưu và nâng cao khác.

## 5. Demo
- Tích hợp giao diện người dùng (ví dụ: Gradio, Streamlit) để trực quan hóa mô hình (khuyến khích).


