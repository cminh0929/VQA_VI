# Vietnamese Visual Question Answering (VQA) - Animals Domain

Dự án xây dựng hệ thống Visual Question Answering (VQA) chuyên biệt cho tiếng Việt, tập trung vào miền dữ liệu **Động vật**. Hệ thống được thiết kế linh hoạt với hai hướng tiếp cận chính: Kiến trúc Modular (CNN + Transformer/LSTM) và Kiến trúc Đa phương thức hiện đại (BLIP + LoRA).

---

## 1. Dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu chuyên biệt về động vật với **3,491 hình ảnh** và **10,800 câu hỏi**.

### 1.1. Cấu trúc thư mục
```text
VQA_VI/
├── data/
│   ├── images/              # Toàn bộ ảnh (.jpg)
│   ├── train_final.json     # 8,640 câu hỏi
│   ├── val_final.json       # 1,043 câu hỏi
│   └── test_final.json      # 1,117 câu hỏi
```

### 1.2. Định dạng JSON
Mỗi mẫu dữ liệu bao gồm ảnh, câu hỏi, danh sách đáp án và nhãn phân loại (Category):
```json
{
    "image_id": "eagle_42.jpg",
    "question": "Đây là con vật gì?",
    "answers": ["chim ưng", "đại bàng"],
    "category": "Object"
}
```
*Các phân loại chính:* `Object`, `Color`, `Action`, `Location`, `Appearance`, `Yes/No`.

---

## 2. Kiến trúc Mô hình (Architecture)

### Hướng A: Kiến trúc Modular (Baseline)
- **Image Encoder**: ResNet-50 (Pretrained).
- **Text Encoder**: PhoBERT (Xử lý tiếng Việt chuyên sâu).
- **Fusion Layer**: Global Fusion kết hợp đặc trưng Ảnh + Văn bản + Loại câu hỏi (Category).
- **Decoder**: LSTM (A1) hoặc Transformer (A2).

### Hướng B: Kiến trúc Đa phương thức (SOTA)
- **Base Model**: `Salesforce/blip-vqa-base`.
- **Fine-tuning**: Sử dụng kỹ thuật **LoRA** (Low-Rank Adaptation) để tối ưu hóa trên dữ liệu tiếng Việt với tài nguyên thấp.

---

## 3. Hướng dẫn sử dụng (Usage)

### 3.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 3.2. Huấn luyện (Training)
* **Modular (Hướng A):**
  ```bash
  python train_modular.py --decoder_type transformer --num_epochs 20
  ```
* **BLIP (Hướng B):**
  ```bash
  python train_blip.py --use_lora True --num_epochs 10
  ```

### 3.3. Đánh giá (Evaluation)
Tính toán các chỉ số Accuracy (Exact Match), BLEU-4 và ROUGE-L trên toàn bộ các model:
```bash
python evaluate_all.py
```

### 3.4. Xem mẫu dự đoán nhanh
Sử dụng script hỗ trợ để xem kết quả dự đoán trực quan trên tập test:
```bash
python scratch/show_predictions.py
```

---

## 4. Kết quả thực nghiệm (Results)

Đánh giá trên tập dữ liệu kiểm thử (**1,117 câu hỏi**):

| Cấu hình | Mô tả | VQA Acc (EM) | BLEU-4 | ROUGE-L |
| :--- | :--- | :---: | :---: | :---: |
| **A1** | Modular + LSTM | 46.91% | 0.0991 | 0.5666 |
| **A2** | Modular + Transformer | **49.15%** | **0.1022** | **0.5903** |
| **B2** | BLIP + LoRA | 3.31%* | 0.0090 | 0.0459 |

*\*Lưu ý: Mô hình BLIP đạt điểm Accuracy thấp do vấn đề thiếu dấu tiếng Việt đầu ra, tuy nhiên khả năng nhận diện hình ảnh rất tốt (kết quả định tính).*

---

## 5. Thư mục và Tập tin quan trọng
- `models/`: Định nghĩa kiến trúc Modular và BLIP.
- `utils/data_loader.py`: Xử lý dữ liệu, phân tách từ (Underthesea) và tăng cường ảnh (Albumentations).
- `config.py`: Quản lý toàn bộ cấu hình hệ thống.
- `doc/`: Chứa các sơ đồ kiến trúc và báo cáo chi tiết.
