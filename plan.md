DỰ ÁN CUỐI KỲ MÔN HỌC SÂU

BÀI 1 (7 điểm): Hệ thống Hỏi đáp trên Ảnh (Visual Question Answering)
Xây dựng hệ thống VQA tiếng Việt trên một miền chuyên biệt. Hệ thống nhận ảnh + câu hỏi tiếng Việt → sinh câu trả lời. Kết hợp kiến thức CNN, LSTM, Transformer và học đa phương thức.
Yêu cầu bắt buộc:
1. Dữ liệu (Thu thập hoặc/và tự xây dựng)
•	Chọn 1 miền chuyên biệt (món ăn Việt, thắng cảnh, biển báo giao thông, nông sản, trang phục truyền thống...).
•	Bộ ba (ảnh, câu hỏi, câu trả lời): 
o	≥ 2000 bộ train (tối thiểu 200 ảnh, mỗi ảnh ≥ 3 câu hỏi).
o	≥ 50 bộ test chuẩn bị thủ công, ảnh không trùng với train.
•	Câu hỏi đa dạng: yes/no, đếm số lượng, nhận dạng, thuộc tính, không gian,... 
•	Câu trả lời ≤ 10 từ.
•	Chia train/val/test 80/10/10.
•	Khuyến khích tăng cường dữ liệu (ảnh: lật/xoay/crop; văn bản: paraphrase, back-translation).
2. Mô hình (bắt buộc cả hai hướng)
Hướng A — Kiến trúc rời:
•	Image encoder: CNN pretrained (ResNet/VGG/EfficientNet) hoặc ViT.
•	Text encoder: LSTM/BiLSTM hoặc PhoBERT.
•	Fusion: concat, element-wise, hoặc co-attention.
•	Answer decoder — bắt buộc so sánh LSTM decoder vs Transformer decoder (giữ nguyên image + text encoder).
Hướng B — Multimodal pretrained:
•	Fine-tune BLIP/BLIP-2, ViLT, LLaVA, Qwen-VL, hoặc PaliGemma (LoRA/PEFT nếu cần).
•	Nêu rõ chiến lược xử lý tiếng Việt (dịch hay dùng trực tiếp).
3. Tìm hiểu phương pháp đánh giá
Chương riêng trong báo cáo phân tích:
•	VQA Accuracy (exact match / soft accuracy chuẩn VQA v2).
•	BLEU, ROUGE-L, METEOR cho câu trả lời dài.
•	BERTScore (ngữ nghĩa).
•	LLM-as-a-judge 
4. Thực nghiệm và so sánh
Bắt buộc 4 cấu hình:
Cấu hình	Mô tả
A1	Hướng A với LSTM decoder
A2	Hướng A với Transformer decoder
B1	Hướng B zero-shot
B2	Hướng B fine-tuned
So sánh A1 vs A2 để làm rõ ảnh hưởng của decoder LSTM vs Transformer.
5. Đánh giá
Đánh gía và so sánh, các tiêu chí dựa vào mục 3.
6. Demo
Khuyến khích có giao diện 
________________________________________
7. Tìm hiểu các giải pháp nâng cao chất lượng mô hình:
1) sử dụng RL
Huấn luyện bổ sung bằng Reinforcement Learning: PPO (reward = VQA Accuracy/BERTScore), DPO, hoặc RLHF.
Yêu cầu tối thiểu:
•	Preference data ≥ 100 cặp.
•	So sánh RL vs SFT trên metric tự động + human eval.
2) Sử dụng các kỹ thuật khác.
 
Sản phẩm nộp
•	Mã nguồn GitHub (README chi tiết).
•	Báo cáo 15–20 trang (có chương đánh giá).
•	Slide + video demo 3–5 phút.
•	Dataset + checkpoint (HuggingFace Hub và Driver).

Bài 2 (3 điểm):

Hãy đề xuất một nhiệm vụ nào đó bạn thấy thú vị liên quan đến giải pháp dùng Deep Learning Model để giải quyết.
Trình bày bài toán, trình bày giải pháp của bạn, code và demo. Phân tích xem tại sao bạn lại đưa giải pháp này.
Nộp bài giống như Bài 1
