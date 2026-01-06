'''
from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
import datetime
import os
import pathlib
from ultralytics import YOLO

app = Flask(__name__)

# ================= CẤU HÌNH =================
VIDEO_PATH = 'traffic_video.mp4'
FOLDER_VI_PHAM = 'hinh_anh_vi_pham'

# TỌA ĐỘ VÙNG CẤM (BẠN CHỈNH LẠI CHO KHỚP VIDEO CỦA BẠN)
STOP_POLYGON = np.array([
    [350, 450], [950, 450], 
    [1200, 720], [100, 720]
], np.int32)

# ================= LOAD MODELS (Chạy 1 lần khi khởi động) =================
print("⏳ Đang tải Model...")
if not os.path.exists(FOLDER_VI_PHAM): os.makedirs(FOLDER_VI_PHAM)

traffic_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
traffic_model.classes = [2, 3, 5, 7]

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
llp_model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector.pt')
ocr_model = YOLO('model/model_nhandien_kytu.pt')
print("✅ Server đã sẵn sàng!")

# ================= BIẾN TOÀN CỤC =================
is_red_light = False  # Trạng thái đèn
known_plates = {}     # Cache biển số

# ================= HÀM HỖ TRỢ (GIỐNG CODE CŨ) =================
def detect_and_ocr_plate(vehicle_img):
    try:
        results = lp_model(vehicle_img)
        df = results.pandas().xyxy[0]
        valid_lp = df[df['confidence'] > 0.5]
        if len(valid_lp) == 0: return None
        row = valid_lp.iloc[0]
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        plate_crop = vehicle_img[y1:y2, x1:x2]
        results_char = ocr_model(plate_crop, verbose=False, conf=0.4)
        chars = []
        for result in results_char:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                char_name = ocr_model.names[cls_id]
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                chars.append({'char': char_name, 'cx': (bx1+bx2)/2, 'cy': (by1+by2)/2})
        if not chars: return None
        height, width = plate_crop.shape[:2]
        if height/width > 0.5:
            avg_cy = sum(c['cy'] for c in chars)/len(chars)
            top = sorted([c for c in chars if c['cy'] < avg_cy], key=lambda x: x['cx'])
            bot = sorted([c for c in chars if c['cy'] >= avg_cy], key=lambda x: x['cx'])
            return "".join([c['char'] for c in top]) + "-" + "".join([c['char'] for c in bot])
        else:
            chars.sort(key=lambda x: x['cx'])
            return "".join([c['char'] for c in chars])
    except: return None

def is_in_stop_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# ================= XỬ LÝ VIDEO STREAM =================
def generate_frames():
    global is_red_light, known_plates
    cap = cv2.VideoCapture(VIDEO_PATH)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            known_plates.clear() # Reset khi hết video
            continue
            
        frame = cv2.resize(frame, (1280, 720))

        # 1. VẼ VÙNG CẤM
        color_zone = (0, 0, 255) if is_red_light else (0, 255, 0)
        cv2.polylines(frame, [STOP_POLYGON], True, color_zone, 2)

        # 2. AI PHÁT HIỆN XE
        results = traffic_model(frame)
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            check_point = (int((x1+x2)/2), y2)
            car_id = f"{x1//30}-{y1//30}"

            is_violation = is_red_light and is_in_stop_zone(check_point, STOP_POLYGON)

            if is_violation:
                # === XE VI PHẠM (CHỈ VẼ KHI VI PHẠM) ===
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                display_text = "DANG DOC..."
                if car_id in known_plates:
                    display_text = known_plates[car_id]
                else:
                    car_img = frame[y1:y2, x1:x2]
                    if car_img.shape[0] > 30:
                        plate_text = detect_and_ocr_plate(car_img)
                        if plate_text:
                            display_text = plate_text
                            known_plates[car_id] = plate_text
                            ts = datetime.datetime.now().strftime('%H%M%S')
                            cv2.imwrite(f"{FOLDER_VI_PHAM}/VIPHAM_{plate_text}_{ts}.jpg", frame)

                (w, h), _ = cv2.getTextSize(display_text, font, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), (0, 0, 255), -1)
                cv2.putText(frame, display_text, (x1, y1 - 5), font, 0.8, (255, 255, 255), 2)

        # Mã hóa ảnh thành bytes để gửi về web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ================= ROUTES (ĐƯỜNG DẪN WEB) =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_light', methods=['POST'])
def toggle_light():
    global is_red_light, known_plates
    is_red_light = not is_red_light
    if not is_red_light:
        known_plates.clear() # Xóa cache khi đèn xanh
    return jsonify({'is_red': is_red_light})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    '''
    
from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
import datetime
import os
import pathlib
from ultralytics import YOLO
import pandas as pd

app = Flask(__name__)

# ================= CẤU HÌNH =================
VIDEO_PATH = 'traffic_video.mp4'
FOLDER_VI_PHAM = 'hinh_anh_vi_pham'

# TỌA ĐỘ VÙNG CẤM (BẠN CHỈNH LẠI CHO KHỚP VIDEO CỦA BẠN)
STOP_POLYGON = np.array([
    [350, 450], [950, 450], 
    [1200, 720], [100, 720]
], np.int32)

# ================= LOAD MODELS =================
print("⏳ Đang tải Model...")
# Tạo thư mục nếu chưa có
if not os.path.exists(FOLDER_VI_PHAM):
    os.makedirs(FOLDER_VI_PHAM)
    print(f"✅ Đã tạo thư mục: {FOLDER_VI_PHAM}")

traffic_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
traffic_model.classes = [2, 3, 5, 7]

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Load model biển số 
lp_model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector.pt')
ocr_model = YOLO('model/model_nhandien_kytu.pt')
print("✅ Server đã sẵn sàng!")

# ================= BIẾN TOÀN CỤC =================
is_red_light = False  # Trạng thái đèn
known_plates = {}     # Cache biển số

# ================= HÀM HỖ TRỢ =================
def detect_and_ocr_plate(vehicle_img):
    try:
        results = lp_model(vehicle_img)
        df = results.pandas().xyxy[0]
        valid_lp = df[df['confidence'] > 0.5]
        if len(valid_lp) == 0: return None
        row = valid_lp.iloc[0]
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        plate_crop = vehicle_img[y1:y2, x1:x2]
        results_char = ocr_model(plate_crop, verbose=False, conf=0.4)
        chars = []
        for result in results_char:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                char_name = ocr_model.names[cls_id]
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                chars.append({'char': char_name, 'cx': (bx1+bx2)/2, 'cy': (by1+by2)/2})
        if not chars: return None
        height, width = plate_crop.shape[:2]
        if height/width > 0.5:
            avg_cy = sum(c['cy'] for c in chars)/len(chars)
            top = sorted([c for c in chars if c['cy'] < avg_cy], key=lambda x: x['cx'])
            bot = sorted([c for c in chars if c['cy'] >= avg_cy], key=lambda x: x['cx'])
            return "".join([c['char'] for c in top]) + "-" + "".join([c['char'] for c in bot])
        else:
            chars.sort(key=lambda x: x['cx'])
            return "".join([c['char'] for c in chars])
    except: return None

def is_in_stop_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# ================= XỬ LÝ VIDEO STREAM =================
def generate_frames():
    global is_red_light, known_plates
    cap = cv2.VideoCapture(VIDEO_PATH)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            known_plates.clear() 
            continue
            
        frame = cv2.resize(frame, (1280, 720))

        # 1. VẼ VÙNG CẤM
        color_zone = (0, 0, 255) if is_red_light else (0, 255, 0)
        cv2.polylines(frame, [STOP_POLYGON], True, color_zone, 2)

        # 2. AI PHÁT HIỆN XE
        results = traffic_model(frame)
        df = results.pandas().xyxy[0]

        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            check_point = (int((x1+x2)/2), y2)
            car_id = f"{x1//30}-{y1//30}" # ID tạm của xe

            is_violation = is_red_light and is_in_stop_zone(check_point, STOP_POLYGON)

            if is_violation:
                # === XE VI PHẠM ===
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                display_text = "Checking..."
                
                # Kiểm tra xem xe này đã xử lý chưa
                if car_id in known_plates:
                    display_text = known_plates[car_id]
                else:
                    # Nếu là xe mới vi phạm -> Xử lý ngay
                    car_img = frame[y1:y2, x1:x2]
                    
                    # 1. Thử đọc biển số
                    plate_text = None
                    if car_img.shape[0] > 30:
                        plate_text = detect_and_ocr_plate(car_img)
                    
                    # 2. Quyết định tên file
                    if plate_text:
                        save_name = plate_text
                        display_text = plate_text
                    else:
                        save_name = "Unknown" # Không đọc được thì đặt là Unknown
                        display_text = "Unknown"

                    # 3. Lưu cache để không xử lý lại xe này
                    known_plates[car_id] = display_text
                    
                    # 4. LƯU ẢNH (QUAN TRỌNG: Lưu bất kể có đọc được hay không)
                    ts = datetime.datetime.now().strftime('%H%M%S')
                    filename = f"VIPHAM_{save_name}_{ts}.jpg"
                    save_path = os.path.join(FOLDER_VI_PHAM, filename)
                    
                    cv2.imwrite(save_path, frame)
                    print(f"📸 Đã lưu ảnh: {save_path}") # Báo ra Terminal để biết

                # Vẽ biển số lên đầu xe
                (w, h), _ = cv2.getTextSize(display_text, font, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), (0, 0, 255), -1)
                cv2.putText(frame, display_text, (x1, y1 - 5), font, 0.8, (255, 255, 255), 2)

        # Gửi ảnh về Web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_light', methods=['POST'])
def toggle_light():
    global is_red_light, known_plates
    is_red_light = not is_red_light
    if not is_red_light:
        known_plates.clear()
    return jsonify({'is_red': is_red_light})

from flask import send_from_directory

# 1. Route để xem trang Lịch Sử
@app.route('/history')
def history():
    # Lấy danh sách tất cả file ảnh trong thư mục vi phạm
    if not os.path.exists(FOLDER_VI_PHAM):
        return render_template('history.html', images=[])
    
    files = os.listdir(FOLDER_VI_PHAM)
    
    # Lọc chỉ lấy file ảnh .jpg và sắp xếp mới nhất lên đầu
    images = [f for f in files if f.endswith('.jpg')]
    images.sort(key=lambda x: os.path.getmtime(os.path.join(FOLDER_VI_PHAM, x)), reverse=True)
    
    return render_template('history.html', images=images)

# 2. Route để web hiển thị được ảnh từ thư mục máy tính
@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(FOLDER_VI_PHAM, filename)

@app.route('/export_excel')
def export_excel():
    # 1. Lấy danh sách file ảnh
    if not os.path.exists(FOLDER_VI_PHAM):
        return "Chưa có dữ liệu!", 404
        
    files = os.listdir(FOLDER_VI_PHAM)
    data = []
    
    # 2. Phân tích tên file để lấy thông tin
    # Tên file: VIPHAM_29A12345_163000.jpg
    for idx, filename in enumerate(files):
        if filename.endswith(".jpg") and "VIPHAM" in filename:
            parts = filename.split("_") # Tách chuỗi
            if len(parts) >= 3:
                bien_so = parts[1]
                time_str = parts[2].replace(".jpg", "")
                # Format lại giờ cho đẹp: 163000 -> 16:30:00
                thoi_gian = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                
                data.append({
                    "STT": idx + 1,
                    "Biển Số": bien_so,
                    "Thời Gian Vi Phạm": thoi_gian,
                    "Tên File Ảnh": filename
                })
    
    # 3. Tạo DataFrame và xuất Excel
    if not data: return "Không tìm thấy dữ liệu vi phạm", 404
    
    df = pd.DataFrame(data)
    excel_path = "Danh_Sach_Vi_Pham.xlsx"
    df.to_excel(excel_path, index=False)
    
    # 4. Gửi file về cho người dùng tải
    return send_from_directory('.', excel_path, as_attachment=True)

# === THÊM API THỐNG KÊ ===
@app.route('/stats')
def stats_page():
    return render_template('stats.html')

@app.route('/api/get_stats')
def get_stats():
    # 1. Quét thư mục ảnh để lấy dữ liệu
    if not os.path.exists(FOLDER_VI_PHAM):
        return jsonify({'total': 0, 'hours': [0]*24})
        
    files = os.listdir(FOLDER_VI_PHAM)
    total = 0
    hours_count = [0] * 24 # Mảng chứa số vi phạm của 24 giờ trong ngày
    
    for f in files:
        if f.endswith('.jpg') and 'VIPHAM' in f:
            total += 1
            # Tách giờ từ tên file: VIPHAM_29A123_163000.jpg
            # Lấy cụm số cuối (163000) -> Lấy 2 số đầu (16) là giờ
            try:
                parts = f.split('_')
                time_part = parts[-1].replace('.jpg', '') # 163000
                hour = int(time_part[:2]) # 16
                if 0 <= hour < 24:
                    hours_count[hour] += 1
            except: pass
            
    return jsonify({
        'total': total,
        'hours': hours_count
    })

if __name__ == "__main__":
    # Tắt reloader để tránh load model 2 lần
    app.run(debug=True, port=5000, use_reloader=False)