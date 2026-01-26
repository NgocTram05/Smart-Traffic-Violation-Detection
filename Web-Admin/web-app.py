from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import pandas as pd
import datetime
import os
import difflib
import numpy as np
from ultralytics import YOLO
import pathlib

app = Flask(__name__)

# --- CẤU HÌNH ---
FILE_CU_DAN = 'cu_dan.csv'
FILE_LICH_SU = 'lich_su_ra_vao.csv'
PATH_DETECT = 'model/LP_detector_nano_61.pt' # Sửa tên model cho đúng máy bạn
PATH_OCR = 'model/model_nhandien_kytu.pt'

# --- BIẾN TOÀN CỤC (Lưu trạng thái hiện tại để gửi ra Web) ---
current_status = {
    "plate": "---",
    "status": "Đang chờ...",
    "color": "secondary", # gray
    "timestamp": "",
    "image_path": "" # Đường dẫn ảnh xe vừa chụp (nếu cần)
}

# --- LOAD MODEL ---
print("⏳ Đang tải Model...")
try:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    model_detect = torch.hub.load('yolov5', 'custom', path=PATH_DETECT, source='local', force_reload=True)
    model_ocr = YOLO(PATH_OCR)
    print("✅ Model AI OK!")
except Exception as e:
    print(f"❌ Lỗi Model: {e}")

# --- LOAD DATA ---
def load_cudan():
    db = {}
    if os.path.exists(FILE_CU_DAN):
        try:
            df = pd.read_csv(FILE_CU_DAN)
            for _, row in df.iterrows():
                clean = str(row['bien_so']).replace('.', '').replace('-', '').replace(' ', '').upper()
                db[clean] = {'name': row['ten_chu_xe'], 'room': row['so_phong']}
        except: pass
    return db

db_cudan = load_cudan()

# --- HÀM XỬ LÝ (Helper) ---
def sap_xep_bien_so(results, height, width):
    # (Copy lại logic sắp xếp biển số cũ của bạn vào đây)
    chars = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            char_name = model_ocr.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            chars.append({'char': char_name, 'cx': (x1+x2)/2, 'cy': (y1+y2)/2})
    if not chars: return ""
    if height/width > 0.5:
        avg_cy = sum(c['cy'] for c in chars)/len(chars)
        top = sorted([c for c in chars if c['cy'] < avg_cy], key=lambda x: x['cx'])
        bot = sorted([c for c in chars if c['cy'] >= avg_cy], key=lambda x: x['cx'])
        return "".join([c['char'] for c in top]) + "-" + "".join([c['char'] for c in bot])
    else:
        chars.sort(key=lambda x: x['cx'])
        return "".join([c['char'] for c in chars])

def log_to_csv(plate, action, person, note):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(FILE_LICH_SU):
        pd.DataFrame(columns=['ThoiGian', 'BienSo', 'HanhDong', 'DoiTuong', 'GhiChu']).to_csv(FILE_LICH_SU, index=False)
    df = pd.DataFrame([{'ThoiGian': now, 'BienSo': plate, 'HanhDong': action, 'DoiTuong': person, 'GhiChu': note}])
    df.to_csv(FILE_LICH_SU, mode='a', header=False, index=False)

def process_frame_logic(frame):
    global current_status
    try:
        # Detect
        results = model_detect(frame)
        df = results.pandas().xyxy[0]
        valid = df[df['confidence'] > 0.6]

        if len(valid) > 0:
            row = valid.iloc[0]
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Vẽ khung
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # OCR
            crop = frame[ymin:ymax, xmin:xmax]
            res_char = model_ocr(crop, verbose=False, conf=0.5)
            text = sap_xep_bien_so(res_char, ymax-ymin, xmax-xmin)

            if len(text) > 4:
                # Logic kiểm tra cư dân
                clean = text.replace('-', '').replace('.', '').replace(' ', '').upper()
                info_text = "KHÁCH VÃNG LAI"
                status_color = "warning" # Vàng (Khách)
                
                # Check DB
                for db_plate, info in db_cudan.items():
                    if difflib.SequenceMatcher(None, clean, db_plate).ratio() > 0.8:
                        info_text = f"CƯ DÂN: {info['name']}"
                        status_color = "success" # Xanh (Cư dân)
                        clean = db_plate
                        break
                
                # Cập nhật biến toàn cục để Web đọc được
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                if current_status["plate"] != clean: # Chỉ log khi biển số thay đổi
                    current_status = {
                        "plate": clean,
                        "status": info_text,
                        "color": status_color,
                        "timestamp": now_str
                    }
                    # Ghi log CSV
                    log_to_csv(clean, "VÀO/RA", info_text, "Tự động")
                    
                # Vẽ chữ lên video
                cv2.putText(frame, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    except Exception as e: pass
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.resize(frame, (800, 480)) # Resize cho nhẹ
        frame = process_frame_logic(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- ROUTES (ĐƯỜNG DẪN WEB) ---
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API: Trả về thông tin xe mới nhất (JSON) để JS cập nhật
@app.route('/api/get_latest')
def get_latest():
    return jsonify(current_status)

# API: Trả về lịch sử (JSON)
@app.route('/api/get_history')
def get_history():
    if os.path.exists(FILE_LICH_SU):
        df = pd.read_csv(FILE_LICH_SU)
        # Lấy 10 dòng cuối, chuyển thành list dictionary
        data = df.tail(10).iloc[::-1].to_dict(orient='records') 
        return jsonify(data)
    return jsonify([])

# API: Mở barie thủ công
@app.route('/api/manual_open', methods=['POST'])
def manual_open():
    # Code gửi tín hiệu xuống Arduino ở đây
    print(">>> MỞ BARIE THỦ CÔNG")
    return jsonify({"success": True, "message": "Đã mở cổng!"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)