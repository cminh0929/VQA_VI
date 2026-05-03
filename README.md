# 🇻🇳 Vietnamese Visual Question Answering (VQA-VN)

Hệ thống Trả lời Câu hỏi trên Ảnh bằng Tiếng Việt (Visual Question Answering) được thiết kế theo cấu trúc module linh hoạt, hỗ trợ cả kiến trúc truyền thống (CNN + LSTM/Transformer) và mô hình đa phương thức hiện đại (PaliGemma - Google).

Dự án này được tối ưu hóa để chạy trên môi trường **Kaggle** và hỗ trợ giao diện thử nghiệm trực quan qua **Gradio**.

---

## 📂 1. Hướng Dẫn Chuẩn Bị Dữ Liệu (Data Setup)

Dự án này được thiết kế theo hướng **Data-Agnostic** (Không phụ thuộc vào một bộ dữ liệu cố định). Bạn hoàn toàn có thể tự thu thập ảnh và tạo câu hỏi của riêng mình (Ví dụ: Nông sản, Biển báo giao thông, Món ăn, v.v.) mà không cần phải thay đổi code của mô hình.

### 1.1. Cấu trúc thư mục dữ liệu chuẩn
Tạo một thư mục tên là `data/` ở thư mục gốc của dự án. Đặt tất cả ảnh và các file JSON vào đúng theo cấu trúc sau:

```text
VQA_VI/
├── data/
│   ├── images/              # (1) BỎ TẤT CẢ ẢNH VÀO ĐÂY (KHÔNG CẦN CHIA THƯ MỤC CON)
│   │   ├── apple_01.jpg
│   │   ├── banana_02.png
│   │   └── ...
│   ├── train.json           # (2) Tệp dữ liệu dùng để Huấn luyện (80%)
│   ├── val.json             # (3) Tệp dữ liệu dùng để Xác thực/Đánh giá (10%)
│   └── test.json            # (4) Tệp dữ liệu Kiểm thử (10%)
```

> **Lưu ý:** Code đã được thiết kế thông minh để tự động tìm ảnh. Bạn chỉ việc gom toàn bộ ảnh vứt thẳng vào thư mục `data/images/`.

### 1.2. Định dạng của tệp JSON
Cả 3 file `train.json`, `val.json`, và `test.json` đều phải tuân thủ nghiêm ngặt cấu trúc mảng JSON gồm các Dictionary như sau:

```json
[
    {
        "image_id": "apple_01.jpg",
        "question": "Trong hình có bao nhiêu quả táo?",
        "answer": "3 quả"
    },
    {
        "image_id": "banana_02.png",
        "question": "Quả chuối có màu gì?",
        "answer": "màu vàng"
    }
]
```
* **`image_id`**: Tên file ảnh (phải khớp chính xác 100% với tên file trong thư mục `data/images/`).
* **`question`**: Câu hỏi bằng tiếng Việt.
* **`answer`**: Câu trả lời ngắn gọn (dưới 10 từ).

---

## 🚀 2. Cài đặt và Môi trường (Setup)

**Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

**Đối với PaliGemma (Hướng B):**
Do `PaliGemma-3B` là mô hình bảo mật của Google, bạn cần phải:
1. Đăng nhập vào [Hugging Face](https://huggingface.co/google/paligemma-3b-pt-224) và bấm nút **"Acknowledge license"** để cấp quyền.
2. Tạo một Access Token (chuỗi mã) tại trang Profile Hugging Face.
3. Nếu chạy trên Kaggle: Thêm Token đó vào mục **Add-ons -> Secrets** với tên `HF_TOKEN`.

---

## 🧠 3. Huấn Luyện Mô Hình (Training)

Hệ thống hỗ trợ 4 phiên bản mô hình khác nhau. Bạn có thể chọn huấn luyện phiên bản nào tùy thích. File lưu trữ (checkpoint) sẽ tự động được tạo trong thư mục `checkpoints/`.

### 3.1. Hướng A: Kiến trúc Module (ResNet50 + PhoBERT)
Được thiết kế xây dựng từ đầu (From scratch) cho phần logic gộp.

* **A1 - LSTM Decoder:**
  ```bash
  python train_modular.py --decoder_type lstm --num_epochs 10
  ```
* **A2 - Transformer Decoder:**
  ```bash
  python train_modular.py --decoder_type transformer --num_epochs 10
  ```

### 3.2. Hướng B: PaliGemma (Google)
Sử dụng mô hình siêu trí tuệ 11GB của Google, tinh chỉnh siêu nhẹ thông qua **LoRA (PEFT)**.

* **B1 - PaliGemma Zero-shot:** (Không cần huấn luyện, chạy thẳng)
* **B2 - PaliGemma Fine-tuned (LoRA):**
  ```bash
  python train_paligemma.py
  ```

---

## 📊 4. Đánh Giá (Evaluation)

Sau khi huấn luyện xong, bạn có thể chạy file đánh giá để so sánh điểm số giữa các mô hình trên tập `val.json`.
Hệ thống sẽ đo đạc bằng 3 thang đo chuẩn xác: **Accuracy**, **BLEU**, và **ROUGE-L**.

```bash
python evaluate_all.py
```
*(Code đánh giá sẽ tự động quét thư mục `checkpoints/` để lấy trọng số mới nhất mà bạn vừa train xong).*

---

## 🎨 5. Chạy Ứng Dụng Demo (Gradio App)

Bạn có thể mở giao diện đồ họa web để tải ảnh lên và thử nghiệm trực tiếp bằng câu hỏi tiếng Việt. Ứng dụng tích hợp sẵn 4 nút bấm tương ứng với 4 mô hình (A1, A2, B1, B2) để bạn dễ dàng đối chiếu sự thông minh của từng mô hình.

```bash
python app.py
```
Sau đó bấm vào đường link `http://127.0.0.1:7860` trên màn hình terminal để sử dụng.
