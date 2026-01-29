# 🚗 SMART TRAFFIC VIOLATION DETECTION SYSTEM
## Hệ thống Nhận diện Biển số và Phát hiện Vi phạm Giao thông Tự động

---

## 📋 MỤC LỤC
1. [Giới thiệu](#giới-thiệu)
2. [Cấu trúc Dự án](#cấu-trúc-dự-án)
3. [Phân tích Hệ thống](#phân-tích-hệ-thống)
4. [Mô tả Chi tiết Các Mô-đun](#mô-tả-chi-tiết-các-mô-đun)
5. [Cách Chạy Hệ thống](#cách-chạy-hệ-thống)
6. [Tính Năng Chi tiết](#tính-năng-chi-tiết)

---

## 🎯 Giới thiệu

**Smart Traffic Violation Detection System** là một hệ thống tự động phát hiện biển số xe và các hành vi vi phạm giao thông sử dụng:
- **YOLOv5** cho phát hiện biển số (License Plate Detection)
- **YOLOv8** cho nhận dạng ký tự (OCR - Optical Character Recognition)
- **OpenCV** cho xử lý ảnh
- **Flask Web** cho giao diện quản lý và theo dõi realtime

Hệ thống cung cấp 2 giao diện chính:
1. **Tkinter Desktop App** (`main.py`) - Ứng dụng nhận diện biển số bằng thủ công từ webcam
2. **Flask Web Admin** (`Web-Admin/web-app.py`) - Ứng dụng web quản lý với tính năng upload ảnh/video, realtime monitoring

---

## 📁 Cấu trúc Dự án

```
Smart-Traffic-Violation-Detection/
│
├── main.py                          # ⭐ Ứng dụng Desktop Tkinter (Nhận diện thủ công)
├── config.py                        # Cấu hình tối ưu cho AI models
├── requirements.txt                 # Danh sách các thư viện cần cài
│
├── function/                        # 📦 Thư viện xử lý hỗ trợ
│   ├── __init__.py
│   ├── helper.py                   # Hàm đọc biển số (read_plate)
│   └── utils_rotate.py             # Hàm deskew - xoay chỉnh ảnh biển số
│
├── model/                           # 🤖 Thư mục model AI
│   ├── LP_detector.pt              # YOLOv5 - Phát hiện biển số
│   └── model_nhandien_kytu.pt      # YOLOv8 - Nhận dạng ký tự
│
├── yolov5/                          # 📚 YOLOv5 Source Code
│   ├── detect.py
│   ├── train.py
│   ├── models/
│   └── utils/
│
├── static/                          # 📸 Thư mục chứa ảnh tải lên
│   └── uploads/
│
├── Web-Admin/                       # 🌐 Ứng dụng Web Flask
│   ├── web-app.py                  # ⭐ Server Flask chính
│   ├── cu_dan.csv                  # Database xe được miễn phạt
│   ├── lich_su_vi_pham.csv         # Lịch sử vi phạm ghi nhận
│   ├── xe_mien_phat.csv            # Danh sách xe được miễn phạt
│   │
│   ├── model/                      # Model AI (copy từ root)
│   │   ├── LP_detector.pt
│   │   └── model_nhandien_kytu.pt
│   │
│   ├── templates/                  # 🎨 HTML Templates
│   │   ├── index.html              # Trang chủ
│   │   ├── dashboard.html          # Bảng điều khiển
│   │   ├── realtime.html           # Monitor realtime từ camera
│   │   ├── upload_image.html       # Upload ảnh để phát hiện
│   │   ├── upload_image_result.html # Hiển thị kết quả ảnh
│   │   ├── upload_video.html       # Upload video
│   │   ├── upload_video_v2.html    # Phiên bản upload video cải tiến
│   │   ├── upload_video_result.html # Hiển thị kết quả xử lý video
│   │   ├── history.html            # Lịch sử phát hiện
│   │   └── stats.html              # Thống kê
│   │
│   ├── static/                     # Tài nguyên web (CSS, JS, uploads)
│   │   └── uploads/
│   │
│   ├── uploads/                    # Thư mục lưu video được xử lý
│   │
│   ├── yolov5/                     # YOLOv5 cho web (copy)
│   │
│   └── __pycache__/                # Cache Python
│
├── lich_su_ra_vao.csv              # Log ra vào hệ thống
└── lich_su_vi_pham.csv             # Lịch sử vi phạm tổng hợp
```

---

## 🧠 Phân tích Hệ thống

### 🔄 Quy trình Nhận diện Biển số (License Plate Recognition Pipeline)

```
INPUT (Ảnh/Video)
    ↓
[1] Phát hiện Biển số (Detection)
    └─→ Model: YOLOv5 Custom (LP_detector.pt)
    └─→ Output: Tọa độ vùng biển số (x1, y1, x2, y2)
    ↓
[2] Cắt vùng Biển số (Crop)
    └─→ Trích xuất phần ảnh chứa biển số
    ↓
[3] Tiền xử lý Ảnh (Preprocessing) - 📌 QUAN TRỌNG
    ├─→ Clahe: Tăng độ tương phản cuc bộ
    ├─→ Sharpening: Làm sắc nét ảnh
    ├─→ Normalize: Chuẩn hóa về 640x640 cho model
    └─→ Giúp cải thiện độ chính xác OCR
    ↓
[4] Nhận dạng Ký tự (OCR)
    ├─→ Model: YOLOv8 (model_nhandien_kytu.pt)
    ├─→ Detect từng ký tự: số (0-9) + chữ (A-Z trừ I,O,Q)
    └─→ Output: Tọa độ + Confidence từng ký tự
    ↓
[5] Sắp xếp Ký tự (Character Sorting)
    ├─→ Xác định số hàng (1 hàng / 2 hàng)
    ├─→ Sắp xếp theo vị trí X (trái → phải)
    ├─→ Nếu 2 hàng: Sắp xếp hàng trên, thêm "-", sắp xếp hàng dưới
    └─→ Output: Chuỗi biển số (VD: "29AA-12345")
    ↓
[6] Xác minh Vi phạm (Violation Detection) - 📌 QUAN TRỌNG
    ├─→ Kiểm tra biển số trong DB xe miễn phạt
    ├─→ Phát hiện: Đèn tín hiệu, Đè vạch, Đè làn, Đậu sai chỗ
    ├─→ Tính điểm vi phạm
    └─→ Output: Danh sách vi phạm (nếu có)
    ↓
OUTPUT (Biển số + Trạng thái)
    ├─→ ✅ Xanh: Không vi phạm
    ├─→ ⚠️ Vàng: Có vi phạm
    └─→ ❌ Đỏ: Không rõ ký tự
```

---

## 📂 Mô tả Chi tiết Các Mô-đun

### 1️⃣ **main.py** - Ứng dụng Desktop Tkinter (Nhận diện Thủ công)

**Vai trò**: Cung cấp giao diện desktop để nhận diện biển số từ webcam theo phương thức thủ công (nhấn SPACE).

#### 🔑 Các hàm chính:

```python
class SimpleLPRApp:
    ├── __init__(root)
    │   └─→ Khởi tạo giao diện Tkinter, load model AI
    │
    ├── load_models()
    │   ├─→ Load YOLOv5 (LP_detector.pt) - Phát hiện biển số
    │   └─→ Load YOLOv8 (model_nhandien_kytu.pt) - Nhận dạng ký tự
    │
    ├── update_camera()
    │   └─→ Cập nhật video từ webcam (30fps) lên giao diện
    │
    ├── trigger_detection(event)
    │   ├─→ Gọi khi người dùng nhấn SPACE
    │   ├─→ Chạy detect_plate() + ocr_process()
    │   └─→ Hiển thị kết quả lên Label
    │
    ├── ocr_process(crop_img) - 📌 QUAN TRỌNG
    │   ├─→ Tiền xử lý ảnh (preprocess_image)
    │   ├─→ Chạy model OCR để detect ký tự
    │   ├─→ Lọc ký tự hợp lệ (0-9, A-Z trừ I,O,Q)
    │   ├─→ Sắp xếp theo vị trí X, Y
    │   └─→ Trả về chuỗi biển số (VD: "29AA-12345")
    │
    ├── preprocess_image(img) - 📌 QUAN TRỌNG
    │   ├─→ Chuyển sang Grayscale
    │   ├─→ Tăng độ tương phản (CLAHE)
    │   ├─→ Làm sắc nét (Sharpening kernel)
    │   └─→ Chuyển lại BGR để YOLO xử lý
    │
    └── is_valid_plate_char(char)
        └─→ Kiểm tra ký tự hợp lệ cho biển số VN
```

#### 📊 Luồng xử lý:
1. **Khởi động**: Load 2 model AI từ `model/` folder
2. **Capture video** từ webcam (0) realtime
3. **Nhấn SPACE**: trigger_detection() được gọi
4. **Detect plate**: Tìm vùng biển số trong frame
5. **OCR**: Đọc từng ký tự trong vùng
6. **Hiển thị**: In biển số + ảnh crop lên GUI

#### 🎨 Giao diện:
- **Video**: Khung hiển thị realtime từ webcam (640x360)
- **Kết quả**: Chuỗi biển số đọc được (28px bold, xanh lá nếu thành công)
- **Ảnh crop**: Hiển thị vùng biển số được cắt ra

---

### 2️⃣ **Web-Admin/web-app.py** - Server Flask (Quản lý Web)

**Vai trò**: Cung cấp giao diện web để upload ảnh/video, monitor realtime, xem lịch sử.

#### 🔑 Các hàm chính:

```python
GLOBAL VARIABLES:
├── current_status
│   ├─→ plate: Biển số đọc được
│   ├─→ status: Trạng thái (vi phạm / không vi phạm)
│   ├─→ color: Màu cảnh báo (success/warning/danger)
│   └─→ timestamp: Thời gian phát hiện
│
└── VIDEO_CONFIG
    ├─→ frame_skip: Bỏ qua N frame (tối ưu tốc độ)
    ├─→ ocr_ensemble_frames: Kết hợp 3 frame liên tiếp
    ├─→ fuzzy_match_threshold: 0.80 (nhận dạng với sai số nhỏ)
    └─→ clarity_threshold: 0.70 (ngưỡng rõ ảnh)

ROUTES (Đường dẫn Web):
├── @app.route('/')
│   └─→ Trang chủ (index.html)
│
├── @app.route('/dashboard')
│   └─→ Bảng điều khiển tổng hợp
│
├── @app.route('/realtime')
│   └─→ Monitor realtime từ camera (gen_frames)
│
├── @app.route('/upload_image', methods=['GET','POST'])
│   ├─→ GET: Hiển thị form upload
│   └─→ POST: Xử lý upload ảnh
│
├── @app.route('/upload_video', methods=['GET','POST'])
│   └─→ Upload & xử lý video
│
├── @app.route('/history')
│   └─→ Xem lịch sử phát hiện
│
└── @app.route('/stats')
    └─→ Thống kê vi phạm

CORE FUNCTIONS:
├── load_cudan() - 📌 QUAN TRỌNG
│   ├─→ Load DB từ 'xe_mien_phat.csv'
│   ├─→ Xây dựng dict: biển_số → ghi chú
│   └─→ Dùng để kiểm tra xe miễn phạt
│
├── sap_xep_bien_so(results, height, width) - 📌 QUAN TRỌNG
│   ├─→ Nhận kết quả OCR (boxes + chars)
│   ├─→ Xác định số hàng (1 hàng / 2 hàng)
│   ├─→ Sắp xếp ký tự theo vị trí X
│   └─→ Trả về: (biển_số_string, avg_confidence)
│
├── detect_violation(frame, plate_box) - 📌 QUAN TRỌNG
│   ├─→ Phát hiện đèn tín hiệu (traffic light):
│   │   └─→ Kiểm tra màu đỏ, gần biển số
│   ├─→ Phát hiện đè vạch (crosswalk):
│   │   └─→ Kiểm tra vị trí biển số vs vạch
│   ├─→ Phát hiện đè làn (lane violation):
│   │   └─→ Kiểm tra kích thước + vị trí biển số
│   └─→ Phát hiện đậu sai (parking violation)
│       └─→ Tìm parking meter gần biển số
│
├── process_frame(frame) - 📌 QUAN TRỌNG
│   ├─→ Detect biển số
│   ├─→ Cắt vùng biển số
│   ├─→ OCR ký tự
│   ├─→ Kiểm tra vi phạm
│   ├─→ Vẽ khung kết quả (xanh/vàng/đỏ)
│   ├─→ Cập nhật current_status
│   └─→ Trả về frame đã vẽ
│
├── gen_frames() - 📌 QUAN TRỌNG
│   ├─→ Generator cho /video_feed (realtime)
│   ├─→ Lặp vô hạn: Capture frame → process → encode JPEG
│   ├─→ Yield frame qua HTTP stream (Motion JPEG)
│   └─→ Dùng cho broadcast realtime
│
├── allowed_file(filename)
│   └─→ Kiểm tra file có phải ảnh/video không
│
├── process_image(image_path) - 📌 QUAN TRỌNG
│   ├─→ Load ảnh từ disk
│   ├─→ Detect + OCR
│   ├─→ Kiểm tra vi phạm
│   ├─→ Ghi log vào CSV
│   └─→ Trả về: (frame_drawn, plate, info, color, crop, clarity)
│
├── process_video(video_path) - 📌 QUAN TRỌNG
│   ├─→ Load video từ disk
│   ├─→ Đọc frame-by-frame (skip để tối ưu)
│   ├─→ Xử lý từng frame với process_frame()
│   ├─→ Kết hợp kết quả 3 frame liên tiếp (ensemble)
│   ├─→ Lọc duplicate plate detection
│   ├─→ Ghi log các phát hiện độc nhất
│   ├─→ Encode video output & lưu
│   └─→ Trả về: danh sách biển số phát hiện
│
└── log_to_csv(plate, status, detail, source)
    └─→ Ghi log vào 'lich_su_vi_pham.csv'
```

#### 📊 Luồng xử lý **Upload Ảnh**:
1. Người dùng chọn file ảnh → POST /upload_image
2. Kiểm tra file hợp lệ → Lưu vào static/uploads/
3. Gọi process_image() → Detect + OCR + Vi phạm
4. Vẽ khung kết quả → Encode ảnh output
5. Ghi log vào CSV → Trả về HTML hiển thị kết quả

#### 📊 Luồng xử lý **Upload Video**:
1. Người dùng chọn file video → POST /upload_video
2. Kiểm tra file hợp lệ → Lưu vào uploads/
3. Gọi process_video() → Xử lý frame-by-frame
4. **Ensemble**: Kết hợp kết quả 3 frame liên tiếp (vote)
5. **De-duplicate**: Chỉ ghi log lần đầu detect biển số
6. Encode video output (MP4) → Lưu & trả về link download

#### 🎯 Các tối ưu hóa **Video Processing**:
```python
VIDEO_CONFIG = {
    'frame_skip': 5,              # Xử lý 1/5 frame (tối ưu tốc độ)
    'ocr_ensemble_frames': 3,     # Vote từ 3 frame → kết quả chắc chắn
    'fuzzy_match_threshold': 0.80 # Chấp nhận sai 20% (nhầm chữ)
}
```
- **Frame Skip**: Video 30fps → xử lý 6fps đủ
- **Ensemble**: Nếu 3 frame liên tiếp đều detect "29AA-12345" → chắc 100%
- **Fuzzy Matching**: "29AA-12345" ≈ "29AA-1234S" (nhầm chữ S thành 5)

---

### 3️⃣ **function/helper.py** - Hàm Đọc Biển số

**Vai trò**: Hàm cũ để đọc biển số (1 hàng / 2 hàng) bằng YOLOv5.

```python
def read_plate(yolo_license_plate, im):
    """
    Đọc biển số từ ảnh sử dụng YOLOv5 OCR model
    
    Input:
        yolo_license_plate: Model YOLO đã load
        im: Ảnh input (OpenCV format)
    
    Output:
        str: Biển số (VD: "29AA-12345" hoặc "29AA12345")
    
    Logic:
        1. Detect ký tự trong ảnh
        2. Nếu < 7 ký tự hoặc > 10: return "unknown"
        3. Tìm điểm trái nhất & phải nhất
        4. Vẽ đường từ trái sang phải
        5. Xác định số hàng (1 hay 2):
           - Nếu tất cả ký tự trên đường = 1 hàng
           - Nếu có ký tự cách xa đường = 2 hàng
        6. Sắp xếp:
           - 1 hàng: sort theo X
           - 2 hàng: sort hàng trên, thêm "-", sort hàng dưới
        7. Ghép chuỗi ký tự
    """
```

---

### 4️⃣ **function/utils_rotate.py** - Xoay chỉnh Ảnh

**Vai trò**: Sửa chữa ảnh bị xiên bằng Deskew (rotation correction).

```python
def changeContrast(img):
    """Tăng độ tương phản sử dụng CLAHE"""
    # Chuyển BGR → LAB
    # CLAHE trên channel L (brightness)
    # Chuyển LAB → BGR
    
def rotate_image(image, angle):
    """Xoay ảnh với góc cho trước"""
    
def compute_skew(src_img, center_thres):
    """
    Tính toán góc xiên của ảnh
    
    Logic:
        1. Canny edge detection
        2. HoughLinesP: Tìm các đường thẳng
        3. Tìm đường nằm cao nhất (min_line)
        4. Tính góc xiên từ đường đó
        5. Trả về góc (degrees)
    """
    
def deskew(src_img, change_cons, center_thres):
    """
    Chỉnh lại ảnh xiên
    
    Input:
        src_img: Ảnh gốc
        change_cons: 1 = tăng contrast trước, 0 = không
        center_thres: 1 = bỏ qua ký tự ở trên cùng
    
    Output:
        Ảnh đã được xoay thẳng
    """
```

---

### 5️⃣ **config.py** - Cấu hình Tối ưu

**Vai trò**: Chứa các thông số tối ưu cho model detect & OCR.

```python
CONFIG = {
    'detector': {
        'conf_threshold': 0.45,    # Chấp nhận detection > 45%
        'iou_threshold': 0.45,     # NMS loại bỏ overlap > 45%
        'img_size': 640,           # Kích thước input YOLOv5
        'max_det': 10,             # Tối đa 10 biển số/frame
    },
    'ocr': {
        'conf_threshold': 0.6,     # Chấp nhận ký tự > 60%
        'iou_threshold': 0.3,      # NMS loại bỏ overlap > 30%
        'img_size': 320,           # Kích thước input OCR
        'max_det': 20,             # Tối đa 20 ký tự/biển số
    },
    'image_processing': {
        'clahe_clip_limit': 3.0,   # Mức clip CLAHE
        'blur_kernel_size': (3,3), # Giảm noise
        'sharpen_kernel': ...       # Làm sắc nét
    }
}
```

---

### 6️⃣ **Model Files**

| File | Model | Vai trò | Input | Output |
|------|-------|--------|-------|--------|
| `model/LP_detector.pt` | YOLOv5 Custom | Phát hiện biển số | Ảnh (640x640) | Tọa độ biển số (x1,y1,x2,y2,conf) |
| `model/model_nhandien_kytu.pt` | YOLOv8 | Nhận dạng ký tự | Ảnh biển số crop | Ký tự + Tọa độ (x1,y1,x2,y2,conf,class) |

---

### 7️⃣ **Templates HTML** - Giao diện Web

```
templates/
├── index.html              # Trang chủ - Menu chính
├── dashboard.html          # 📊 Bảng điều khiển (thống kê realtime)
├── realtime.html           # 📹 Monitor camera (streaming video)
├── upload_image.html       # 🖼️ Form upload ảnh
├── upload_image_result.html # Hiển thị kết quả detect ảnh
├── upload_video.html       # 🎬 Form upload video
├── upload_video_v2.html    # 🎬 Version 2 upload video (cải tiến)
├── upload_video_result.html # Hiển thị kết quả detect video
├── history.html            # 📋 Lịch sử phát hiện
└── stats.html              # 📈 Thống kê & biểu đồ
```

---

## 🚀 Cách Chạy Hệ thống

### **Phần 1: Chuẩn bị Môi trường**

#### 1.1. Cài đặt Python & Thư viện

```bash
# Tạo Virtual Environment (nếu chưa có)
python -m venv .venv

# Kích hoạt (Windows)
.\.venv\Scripts\Activate.ps1

# Kích hoạt (Mac/Linux)
source .venv/bin/activate

# Cài đặt thư viện từ requirements.txt
pip install -r requirements.txt

# Cài thêm Flask (nếu cần)
pip install flask
```

#### 1.2. Kiểm tra File Model

```bash
# Model phải tồn tại:
model/LP_detector.pt              # YOLOv5 detect
model/model_nhandien_kytu.pt      # YOLOv8 OCR

# Copy sang Web-Admin (nếu cần):
copy model\*.pt Web-Admin\model\
```

---

### **Phần 2: Chạy Ứng dụng Desktop (main.py)**

#### 2.1. Chạy ứng dụng

```bash
# Từ thư mục root
python main.py
```

#### 2.2. Sử dụng

1. **Cửa sổ sẽ hiển thị** video từ webcam (live)
2. **Bấm phím SPACE** để trigger detection
3. **Kết quả sẽ hiển thị**:
   - 📝 Chuỗi biển số (xanh lá = thành công)
   - 🖼️ Ảnh cắt của vùng biển số
4. **Đóng cửa sổ** để thoát

#### 2.3. Khắc phục sự cố

```python
# Nếu không tìm thấy camera, sửa trong main.py:
self.cap = cv2.VideoCapture(0)  # 0 = webcam mặc định
                                 # 1 = camera thứ 2
                                 # 2 = camera thứ 3...

# Nếu model không tìm thấy:
- Kiểm tra đường dẫn: model/LP_detector.pt
- Kiểm tra file .pt có tồn tại không
- Kiểm tra quyền đọc file
```

---

### **Phần 3: Chạy Ứng dụng Web (Web-Admin/web-app.py)**

#### 3.1. Chạy server Flask

```bash
# Vào thư mục Web-Admin
cd Web-Admin

# Chạy server
python web-app.py

# Hoặc chạy với debug mode
FLASK_ENV=development FLASK_DEBUG=1 python web-app.py
```

#### 3.2. Truy cập Web

```
🌐 Mở trình duyệt (Chrome/Firefox/Edge):
   http://localhost:5000
   
   Nếu lỗi port 5000, sửa trong web-app.py:
   app.run(host='0.0.0.0', port=8080, debug=True)
```

#### 3.3. Các Tính năng Web

**📊 Dashboard** (`/dashboard`)
- Hiển thị tổng số biển số phát hiện
- Số vi phạm ghi nhận
- Xe được miễn phạt
- Biểu đồ thống kê

**📹 Realtime** (`/realtime`)
- Monitor camera trực tiếp
- Hiển thị biển số & trạng thái (xanh/vàng/đỏ)
- Cập nhật realtime

**🖼️ Upload Ảnh** (`/upload_image`)
```
1. Chọn file ảnh (PNG, JPG, JPEG)
2. Bấm "Upload"
3. Hệ thống detect + OCR
4. Hiển thị kết quả:
   - Ảnh đã vẽ khung
   - Biển số đọc được
   - Trạng thái (vi phạm / không)
   - Mức rõ ảnh (Rõ / Mờ)
```

**🎬 Upload Video** (`/upload_video`)
```
1. Chọn file video (MP4, AVI, MOV)
2. Bấm "Upload & Process"
3. Server xử lý video:
   - Đọc frame-by-frame
   - Skip 5 frame để tối ưu
   - Ensemble kết quả 3 frame
   - Ghi log phát hiện độc nhất
4. Download video output (đã vẽ khung)
5. Xem danh sách biển số phát hiện
```

**📋 History** (`/history`)
- Xem tất cả vi phạm ghi nhận
- Filter theo biển số / ngày / loại vi phạm
- Export CSV

**📈 Stats** (`/stats`)
- Biểu đồ thống kê
- Top xe vi phạm nhất
- Loại vi phạm phổ biến
- Thời gian cao điểm

---

### **Phần 4: Data Files**

#### 4.1. CSV Database

**`xe_mien_phat.csv`** - Danh sách xe miễn phạt
```csv
bien_so,ghi_chu
29AA12345,Xe công vụ
29BB45678,Xe cấp bách
```

**`lich_su_vi_pham.csv`** - Lịch sử vi phạm (tự động ghi)
```csv
bien_so,status,detail,timestamp,source
29AA12345,PHÁT HIỆN VI PHẠM,Đi sai đèn tín hiệu,2024-01-29 14:30:45,upload_image
29BB45678,KHÔNG VI PHẠM,OK,2024-01-29 14:31:20,realtime
```

**`lich_su_ra_vao.csv`** - Log ra vào hệ thống

#### 4.2. Thay đổi Database

- Edit trực tiếp file `.csv` bằng Excel / VSCode
- Hoặc thêm route Flask để CRUD qua Web
- Server sẽ reload tự động lần tới

---

## ⚙️ Tính Năng Chi tiết

### 🎯 **Phát hiện Vi phạm**

```python
detect_violation(frame, plate_box) detects:

1. ❌ ĐI SAI ĐÈN TÍN HIỆU (Đèn đỏ)
   ├─→ Phát hiện traffic light
   ├─→ Kiểm tra màu đỏ (RGB)
   └─→ Kiểm tra khoảng cách < 200px từ biển số

2. ❌ ĐE VẠC (Crosswalk)
   ├─→ Phát hiện person / stop sign
   └─→ Kiểm tra vị trí biển số vs vạch

3. ❌ ĐE LÀN ĐƯỜNG
   ├─→ Kiểm tra kích thước biển số
   └─→ Kiểm tra vị trí ở giữa frame

4. ❌ ĐẬU SAI NƠI QUY ĐỊNH
   └─→ Phát hiện parking meter gần biển số
```

### 📊 **Ensemble OCR (Video)**

```python
Process video với 3 frame liên tiếp:

Frame 1: Detect "29AA-12345"  ✓
Frame 2: Detect "29AA-12345"  ✓ (0.95 confidence)
Frame 3: Detect "29AA-12345"  ✓ (0.92 confidence)
                    ↓
          VOTE: 3/3 → 100% chắc chắn
          
          Ghi log 1 lần duy nhất
```

### 🔍 **Fuzzy Matching (Chữ sai lệch)**

```python
fuzzy_match_threshold = 0.80

Biển số ghi: "29AA-12345"
Lần 1:  "29AA-1234S"  (nhầm 5→S) → 90% match → không log
Lần 2:  "29AA-1234Z"  (nhầm 5→Z) → 85% match → không log
Lần 3:  "29AA-12345"  (đúng)     → 100% match → LOG!
```

### 🎨 **Màu cảnh báo**

| Màu | Ý nghĩa | RGB | Hex |
|-----|---------|-----|-----|
| 🟢 Xanh | Không vi phạm | (0,255,0) | #00FF00 |
| 🟡 Vàng | Có vi phạm | (0,255,255) | #FFFF00 |
| 🔴 Đỏ | Không rõ / Lỗi | (0,0,255) | #FF0000 |

---

## 🐛 Debugging & Log

### Xem Log của Server

```bash
# Terminal sẽ in ra:
⏳ Đang tải Model...
✅ Model AI OK!
[29 Jan 2024 14:30:45] Đang chờ...
[29 Jan 2024 14:30:50] Detect: 29AA-12345
[29 Jan 2024 14:30:51] Status: Không vi phạm
```

### Thay đổi Level Log

```python
# Trong web-app.py
import logging
logging.basicConfig(level=logging.DEBUG)  # DEBUG / INFO / WARNING / ERROR
```

---

## 📦 Yêu cầu Hệ thống

| Thành phần | Yêu cầu tối thiểu |
|-----------|-----------------|
| Python | 3.8+ |
| RAM | 4GB (6GB khuyến cáo) |
| GPU (tùy chọn) | NVIDIA CUDA 11.0+ (tăng tốc 10x) |
| Webcam | Độ phân giải 1280x720 trở lên |

---

## 🔧 Tối ưu Hiệu suất

### Tăng tốc GPU (nếu có NVIDIA card)

```python
# Trong config.py hoặc web-app.py
import torch
print(torch.cuda.is_available())  # True = GPU khả dụng

# Models sẽ tự động dùng GPU
# Tốc độ: CPU ~20fps → GPU ~60fps
```

### Giảm độ trễ Video

```python
# Trong web-app.py
VIDEO_CONFIG = {
    'frame_skip': 10,  # Giảm từ 5 → 10 (xử lý 1/10 frame)
    'output_video_fps': 15,  # Giảm từ 30 → 15 FPS output
}
```

---

## 📞 Hỗ trợ & Liên hệ

- **Lỗi Model**: Kiểm tra file `.pt` tồn tại & có quyền đọc
- **Lỗi Camera**: Kiểm tra camera không bị ứng dụng khác sử dụng
- **Lỗi Flask**: Kiểm tra port 5000 không bị dùng → sửa thành 8080
- **Vi phạm không phát hiện**: Tăng `conf_threshold` xuống (0.4)

---

## 📄 Tài liệu Thêm

- **YOLOv5 Docs**: https://docs.ultralytics.com/yolov5/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Flask Docs**: https://flask.palletsprojects.com/
- **OpenCV Docs**: https://docs.opencv.org/

---
