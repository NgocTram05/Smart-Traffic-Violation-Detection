# 🚀 Hướng Dẫn Nhanh - Hệ Thống Upload Video

## ⚡ Bắt Đầu Nhanh (2 phút)

### 1️⃣ Khởi Động Server

```bash
# Mở PowerShell, vào thư mục Web-Admin
cd Web-Admin

# Kích hoạt virtual environment
.\.venv\Scripts\Activate.ps1

# Chạy Flask app
python web-app.py
```

**Kỳ vọng thấy:**
```
✅ Model AI OK!
 * Running on http://127.0.0.1:5000
```

---

### 2️⃣ Truy Cập Trang Upload Video

1. Mở trình duyệt
2. Vào: **http://localhost:5000/upload_video**
3. Sẵn sàng upload video!

---

## 📹 Sử Dụng Trang Upload Video

### Quy Trình Upload & Xử Lý

```
Chọn File
    ↓
Kiểm tra (File size, định dạng)
    ↓
Preview video
    ↓
Click "Run"
    ↓
Upload & Xử lý
    ↓
Hishow mức tiến độ (Progress bar)
    ↓
Kết quả: Danh sách biển số
```

### Chi Tiết Từng Bước

#### **Bước 1: Chọn File**
- Click **"Chọn File"** hoặc kéo thả file vào
- Hỗ trợ: **MP4, AVI, MOV**
- Giới hạn: **≤500MB**

```
❌ File quá lớn? 
   → Nén video hoặc cắt ngắn
   
❌ Định dạng sai?
   → Chuyển đổi sang MP4
```

#### **Bước 2: Xem Trước**
- Video tự động hiển thị trong hộp **"Bảng Giám Sát Video"**
- Có controls: play, pause, seek

```
💡 Mẹo: Xem trước trước khi submit để chắc chắn
```

#### **Bước 3: Chọn "Run"**
- Nút **"Run"** bật lên sau khi chọn file hợp lệ
- Click để bắt đầu xử lý

#### **Bước 4: Theo Dõi Tiến Độ**
- **Progress Bar** hiển thị tiến độ (0-100%)
- Thông báo: "Đang upload và xử lý video..."
- **Không** tắt tab hay reload trong quá trình xử lý!

#### **Bước 5: Xem Kết Quả**
```
Hiển thị 3 loại status:
✓ Bình thường    (xanh) - Chính chủ
⚠️ Vi phạm       (vàng) - Phát hiện vi phạm
❌ Mờ/Không rõ   (đỏ)   - Ảnh mờ, không nhận diện
```

Mỗi kết quả hiển thị:
- 📷 **Ảnh crop biển số**
- 🚗 **Loại xe** (Xe máy, Ô tô, Xe bus)
- ⏱️ **Thời điểm** (giây trong video)
- 👁️ **Độ rõ** (Rõ / Mờ)
- 📝 **Chi tiết** (Loại xe - Biển số - Trạng thái)
- 💾 **Nút lưu** vào lịch sử

---

## 🎯 Ví Dụ Thực Tế

### Video: demo.mp4 (30 FPS, 16.9 giây)

**Đầu vào:**
- File: `demo.mp4`
- Kích thước: 145 MB ✓
- Định dạng: MP4 ✓

**Tiến độ:**
```
Đang xử lý... 10%
Đang xử lý... 45%
Đang xử lý... 90%
✓ Hoàn tất! 100%
```

**Kết quả:** (Ví dụ)
```
✓ Phát hiện 3 biển số không trùng lặp

1. ✓ BIỂN SỐ: 29A-123456
   Loại xe: Ô tô | Thời điểm: 5.2s | Độ rõ: Rõ
   Chi tiết: Ô tô - 29A-123456 - Bình thường
   [Ảnh crop] [Lưu vào lịch sử]

2. ⚠️ BIỂN SỐ: 30B-789012
   Loại xe: Xe máy | Thời điểm: 8.7s | Độ rõ: Rõ
   Chi tiết: Xe máy - 30B-789012 - Vi phạm: Đi sai đèn tín hiệu
   [Ảnh crop] [Lưu vào lịch sử]

3. ❌ BIỂN SỐ: 36K-345678
   Loại xe: Ô tô | Thời điểm: 12.3s | Độ rõ: Mờ
   Chi tiết: Ô tô - Hình ảnh mờ, không rõ
   [Ảnh crop] [Lưu vào lịch sử]
```

---

## ⚙️ Cấu Hình & Tùy Chỉnh

### VIDEO_CONFIG trong `web-app.py`

```python
VIDEO_CONFIG = {
    'frame_skip': 5,                    # Xử lý mỗi N frame (tăng = nhanh hơn)
    'plate_confidence_threshold': 0.60, # Ngưỡng phát hiện biển số
    'clarity_threshold': 0.70,          # Ngưỡng đánh giá độ rõ
}
```

**Ví dụ điều chỉnh:**
- Nhanh hơn? → `'frame_skip': 10`
- Chính xác hơn? → `'plate_confidence_threshold': 0.75`

---

## 🐛 Troubleshooting

### ❌ "File quá lớn (775 MB). Tối đa 500MB"

**Giải pháp:**
1. Cắt video thành đoạn nhỏ hơn
2. Hoặc nén video:
   ```bash
   ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
   ```

### ❌ "Định dạng video không hợp lệ"

**Giải pháp:**
- Hỗ trợ: **MP4, AVI, MOV**
- Chuyển đổi sang MP4:
  ```bash
  ffmpeg -i video.mov output.mp4
  ```

### ❌ "Không tìm thấy biển số trong video"

**Giải pháp:**
1. Video quá mờ hoặc góc lạ?
2. Biển số quá nhỏ hoặc xa?
3. Thử điều chỉnh `'plate_confidence_threshold'` xuống 0.5

### ❌ Server báo lỗi

**Kiểm tra:**
```bash
# Xem log chi tiết
tail -f Web-Admin/upload_video.log

# Hoặc mở file trực tiếp
Web-Admin/upload_video.log
```

---

## 🔐 Bảo Mật & Best Practice

### ✅ Làm
- ✅ Upload video từ nguồn tin cậy
- ✅ Kiểm tra file trước khi submit
- ✅ Không upload video nhạy cảm

### ❌ Không Làm
- ❌ Không tắt tab trong quá trình xử lý
- ❌ Không upload file lớn (>500MB)
- ❌ Không reload page khi đang xử lý

---

## 📊 Xem Lịch Sử

1. Click **"Lịch Sử"** trong menu
2. Xview tất cả kết quả phát hiện:
   - Thời gian
   - Biển số
   - Hành động
   - Chi tiết

```bash
# Hoặc xem file CSV trực tiếp
Web-Admin/lich_su_vi_pham.csv
```

---

## 📱 Các Trang Khác

| Trang | Chức Năng |
|-------|----------|
| **/** | Dashboard - Cập nhật realtime |
| **/upload_image** | Upload ảnh tĩnh |
| **/upload_video** | Upload video (đây) |
| **/history** | Xem lịch sử phát hiện |
| **/stats** | Thống kê |

---

## 🆘 Hỗ Trợ

Nếu có vấn đề:

1. **Kiểm tra kết nối server**
   ```bash
   Truy cập: http://localhost:5000
   ```

2. **Xem log chi tiết**
   ```bash
   Web-Admin/upload_video.log
   ```

3. **Restart server**
   ```bash
   Ctrl+C (dừng) → python web-app.py (chạy lại)
   ```

---

**Cập nhật: 08/02/2026** | **Phiên bản: 2.0**

Bây giờ bạn đã sẵn sàng! 🎉
