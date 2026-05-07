# Vietnamese Visual Question Answering (VQA)

Hệ thống Visual Question Answering (VQA) dành riêng cho tiếng Việt. Dự án được phát triển dựa trên cấu trúc linh hoạt (modular design), cho phép triển khai cả kiến trúc rời rạc (CNN + LSTM/Transformer) và kiến trúc học đa phương thức hiện đại (PaliGemma) trên các tập dữ liệu tùy chỉnh.

Hệ thống được tối ưu hóa để vận hành trên môi trường Kaggle Notebook và cung cấp giao diện tương tác (Gradio) phục vụ mục đích kiểm thử cục bộ.

---

## 1. Hướng Dẫn Chuẩn Bị Dữ Liệu

Kiến trúc của dự án là Data-Agnostic (không phụ thuộc vào một miền dữ liệu cụ thể). Người dùng có thể sử dụng dữ liệu thuộc bất kỳ miền nào (nông sản, giao thông, y tế,...) bằng cách tuân thủ cấu trúc thư mục và định dạng tệp chuẩn dưới đây.

### 1.1. Cấu trúc thư mục dữ liệu chuẩn

Người dùng cần khởi tạo thư mục `data/` tại thư mục gốc của dự án. Tất cả hình ảnh và tệp nhãn (JSON) phải được tổ chức theo sơ đồ sau:

```text
VQA_VI/
├── data/
│   ├── images/              
│   │   ├── image_001.jpg
│   │   ├── image_002.png
│   │   └── ...
│   ├── train.json           # Dữ liệu huấn luyện (khuyến nghị 80%)
│   ├── val.json             # Dữ liệu xác thực (khuyến nghị 10%)
│   └── test.json            # Dữ liệu kiểm thử (khuyến nghị 10%)
```

**Lưu ý:** Hệ thống có khả năng tự động nội suy đường dẫn tệp ảnh. Người dùng không cần phân chia ảnh thành các thư mục con phức tạp; toàn bộ tệp tin hình ảnh có thể được gom chung vào thư mục `data/images/`.

### 1.2. Định dạng cấu trúc tệp JSON

Các tệp `train.json`, `val.json`, và `test.json` phải tuân thủ nghiêm ngặt định dạng danh sách (Array) chứa các đối tượng từ điển (Dictionary) như ví dụ sau:

```json
[
    {
        "image_id": "image_001.jpg",
        "question": "Câu hỏi bằng tiếng Việt có cấu trúc rõ ràng?",
        "answer": "câu trả lời ngắn gọn"
    },
    {
        "image_id": "image_002.png",
        "question": "Màu sắc của đối tượng trong hình là gì?",
        "answer": "màu đỏ"
    }
]
```

* **`image_id`**: Định danh của hình ảnh, bắt buộc phải trùng khớp hoàn toàn (kể cả phần mở rộng tệp) với tên tệp vật lý lưu trong thư mục `data/images/`.
* **`question`**: Câu truy vấn bằng tiếng Việt liên quan đến nội dung bức ảnh.
* **`answer`**: Nhãn thực tế (Ground truth) để mô hình học tập, ưu tiên các câu trả lời ngắn gọn gọn (dưới 10 từ).

---

## 2. Thiết Lập Môi Trường (Setup)

**Cài đặt các gói phụ thuộc:**
```bash
pip install -r requirements.txt
```

**Cấp quyền đối với kiến trúc PaliGemma:**
Kiến trúc PaliGemma (Phiên bản B) thuộc diện mô hình bảo mật (Gated Model) do Google quản lý. Người dùng cần thực hiện các bước sau trước khi tiến hành huấn luyện:
1. Truy cập [Hugging Face PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) và chấp thuận điều khoản sử dụng (Acknowledge license).
2. Khởi tạo Access Token tại mục Settings của tài khoản Hugging Face cá nhân.
3. Nếu sử dụng Kaggle: Cấu hình biến môi trường bằng cách thêm Access Token vào mục **Add-ons -> Secrets** với định danh là `HF_TOKEN`.

---

## 3. Huấn Luyện Mô Hình (Training)

Hệ thống hỗ trợ 4 phiên bản mô hình khác nhau theo kế hoạch thực nghiệm ban đầu. Kết quả huấn luyện (checkpoint) sẽ tự động được ghi nhận tại thư mục `checkpoints/`.

### 3.1. Hướng A: Kiến trúc Module (ResNet50 + PhoBERT)
Kiến trúc này tách rời quá trình trích xuất đặc trưng hình ảnh và văn bản trước khi thực hiện cơ chế kết hợp (Fusion).

* **Mô hình A1 - LSTM Decoder:**
  ```bash
  python train_modular.py --decoder_type lstm --num_epochs 10
  ```
* **Mô hình A2 - Transformer Decoder:**
  ```bash
  python train_modular.py --decoder_type transformer --num_epochs 10
  ```



---

## 4. Đánh Giá Hiệu Suất (Evaluation)

Sau khi hoàn tất quá trình huấn luyện, người dùng có thể kích hoạt tập lệnh đánh giá tự động trên tập dữ liệu `val.json`.
Hệ thống sử dụng 3 bộ tiêu chuẩn đo lường phổ biến: **VQA Accuracy**, **BLEU**, và **ROUGE-L**.

```bash
python evaluate_all.py
```
*(Tập lệnh sẽ tự động xác định và tải tệp trọng số mới nhất từ thư mục `checkpoints/` đối với từng cấu trúc mô hình).*

---

## 5. Triển Khai Giao Diện Thử Nghiệm (Inference)

Giao diện tương tác trực tiếp được xây dựng bằng Gradio, cho phép tải lên hình ảnh vật lý và đặt câu hỏi trực tiếp để đối chiếu chất lượng sinh văn bản giữa các mô hình.

```bash
python app.py
```
Sau khi khởi động dịch vụ thành công, truy cập `http://127.0.0.1:7860` thông qua trình duyệt web để bắt đầu phiên thử nghiệm.
