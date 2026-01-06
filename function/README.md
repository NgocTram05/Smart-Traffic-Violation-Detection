# 📸 HỆ THỐNG PHÁT HIỆN VƯỢT ĐÈN ĐỎ & NHẬN DIỆN BIỂN SỐ (Smart Traffic Violation Detection)

## 📖 Giới thiệu
Đây là hệ thống giám sát giao thông thông minh sử dụng Trí tuệ nhân tạo (AI) để tự động phát hiện các phương tiện vượt đèn đỏ. Hệ thống bao gồm giao diện Web trực quan, cho phép xem video trực tiếp (Live stream), xem lại lịch sử vi phạm và xuất báo cáo.

Dự án sử dụng **YOLOv5** để nhận diện phương tiện, **Custom Model** để bắt biển số và **OCR** để đọc ký tự.

## 🚀 Tính năng chính
* ✅ **Giám sát thời gian thực:** Phát hiện xe máy, ô tô, xe buýt, xe tải.
* ✅ **Bắt lỗi vượt đèn đỏ:** Tự động chụp ảnh xe đi vào vùng cấm khi đèn đỏ.
* ✅ **Nhận diện biển số (ALPR):** Đọc và hiển thị biển số xe vi phạm ngay trên màn hình.
* ✅ **Điều khiển đèn tín hiệu:** Giả lập bật/tắt đèn xanh đỏ (phím Space).
* ✅ **Giao diện Web (Dashboard):**
    * Xem Camera trực tiếp.
    * Xem lại Lịch sử vi phạm (có ảnh bằng chứng).
    * Biểu đồ Thống kê vi phạm theo khung giờ.
* ✅ **Báo cáo:** Xuất danh sách vi phạm ra file Excel (`.xlsx`).

## 🛠️ Công nghệ sử dụng
* **Ngôn ngữ:** Python 3.9+
* **Framework Web:** Flask
* **Computer Vision:** OpenCV, PyTorch
* **AI Models:**
    * Nhận diện xe: YOLOv5s (Pretrained)
    * Phát hiện biển số: `LP_detector_nano_61.pt`
    * Đọc ký tự (OCR): `model_nhandien_kytu.pt`
* **Frontend:** HTML5, Bootstrap 5, Chart.js

## 📂 Cấu trúc dự án
```text
Smart-Parking/
├── model/                  # Chứa các file model AI (.pt)
├── templates/              # Giao diện Web (HTML)
│   ├── index.html          # Trang chủ (Live View)
│   ├── history.html        # Trang lịch sử & Xuất Excel
│   └── stats.html          # Trang thống kê biểu đồ
├── hinh_anh_vi_pham/       # Thư mục lưu ảnh bằng chứng (Tự động tạo)
├── web-app.py                  # Code xử lý chính (Flask Server)
├── requirements.txt        # Danh sách thư viện cần cài đặt
└── traffic_video.mp4       # Video đầu vào để test

⚙️ Hướng dẫn cài đặt
Bước 1: Chuẩn bị môi trường
Đảm bảo máy tính đã cài Python. Mở Terminal và chạy lệnh sau để cài đặt các thư viện cần thiết:

Bash

pip install -r requirements.txt
(File requirements.txt bao gồm: flask, opencv-python, torch, pandas, openpyxl, ultralytics, ...)

Bước 2: Cấu hình Vùng Cấm (Quan trọng)
Mở file web-app.py, tìm đến dòng STOP_POLYGON và chỉnh sửa 4 tọa độ [x, y] sao cho khớp với vạch dừng trong video của bạn:

Python

STOP_POLYGON = np.array([
    [350, 450], [950, 450], 
    [1200, 720], [100, 720]
], np.int32)
▶️ Hướng dẫn sử dụng
Chạy hệ thống:

Bash

python web-app.py
Truy cập Web: Mở trình duyệt và vào địa chỉ: http://localhost:5000

Thao tác:

Nhấn phím SPACE (Cách) trên bàn phím (hoặc nút trên web) để chuyển đổi đèn XANH / ĐỎ.

Khi đèn ĐỎ, xe đi vào vùng cấm sẽ bị chụp ảnh và lưu vào Lịch sử.

Vào tab "Lịch Sử" để xem lại và tải file Excel.

Vào tab "Thống Kê" để xem biểu đồ.

