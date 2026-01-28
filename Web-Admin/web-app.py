from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, send_file
import cv2
import torch
import pandas as pd
import datetime
import os
import difflib
import numpy as np
from ultralytics import YOLO
import pathlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='werkzeug')
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
from werkzeug.utils import secure_filename
import io
import traceback

app = Flask(__name__)

# --- CẤU HÌNH ---
FILE_CU_DAN = 'xe_mien_phat.csv'
FILE_LICH_SU = 'lich_su_vi_pham.csv'
PATH_DETECT = 'model/LP_detector.pt' # Sửa tên model cho đúng
PATH_OCR = 'model/model_nhandien_kytu.pt'

# Cấu hình upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- BIẾN TOÀN CỤC (Lưu trạng thái hiện tại để gửi ra Web) ---
current_status = {
    "plate": "---",
    "status": "Đang chờ...",
    "color": "secondary", # gray
    "timestamp": "",
    "image_path": "" # Đường dẫn ảnh xe vừa chụp (nếu cần)
}

# Camera cho realtime
camera = None
is_running = True

# --- LOAD MODEL ---
print("⏳ Đang tải Model...")
try:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH_DETECT, force_reload=True)
    model_ocr = YOLO(PATH_OCR)
    model_general = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
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
                db[clean] = {'note': row['ghi_chu']}
        except: pass
    return db

db_cudan = load_cudan()

# --- HÀM XỬ LÝ (Helper) ---
def sap_xep_bien_so(results, height, width):
    # (Copy lại logic sắp xếp biển số cũ của bạn vào đây)
    chars = []
    total_conf = 0
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            char_name = model_ocr.names[cls_id]
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            chars.append({'char': char_name, 'cx': (x1+x2)/2, 'cy': (y1+y2)/2, 'conf': conf})
            total_conf += conf
    avg_conf = total_conf / len(chars) if chars else 0
    if not chars: return "", 0
    if height/width > 0.5:
        avg_cy = sum(c['cy'] for c in chars)/len(chars)
        top = sorted([c for c in chars if c['cy'] < avg_cy], key=lambda x: x['cx'])
        bot = sorted([c for c in chars if c['cy'] >= avg_cy], key=lambda x: x['cx'])
        return "".join([c['char'] for c in top]) + "-" + "".join([c['char'] for c in bot]), avg_conf
    else:
        chars.sort(key=lambda x: x['cx'])
        return "".join([c['char'] for c in chars]), avg_conf

def detect_violation(frame, plate_box):
    """Phát hiện vi phạm giao thông dựa trên frame và vị trí biển số"""
    violations = []
    try:
        results = model_general(frame)
        df = results.pandas().xyxy[0]
        
        px1, py1, px2, py2 = plate_box
        plate_center_x = (px1 + px2) / 2
        plate_center_y = (py1 + py2) / 2
        
        # Phát hiện đèn tín hiệu
        traffic_lights = df[df['name'] == 'traffic light']
        for _, tl in traffic_lights.iterrows():
            tl_x1, tl_y1, tl_x2, tl_y2 = tl['xmin'], tl['ymin'], tl['xmax'], tl['ymax']
            # Kiểm tra nếu đèn gần biển số (trong khoảng cách hợp lý)
            if abs(tl_x1 - px1) < 200 and abs(tl_y1 - py1) < 200:  # Khoảng cách pixel
                # Crop vùng đèn và kiểm tra màu
                tl_crop = frame[int(tl_y1):int(tl_y2), int(tl_x1):int(tl_x2)]
                if tl_crop.size > 0:
                    avg_color = cv2.mean(tl_crop)[:3]  # BGR
                    r, g, b = avg_color[2], avg_color[1], avg_color[0]  # RGB
                    if r > g + 30 and r > b + 30:  # Đỏ
                        violations.append("Đi sai đèn tín hiệu (đèn đỏ)")
        
        # Phát hiện đè vạch (crosswalk hoặc stop sign)
        crosswalks = df[df['name'].isin(['person', 'stop sign'])]  # Giả định person ở crosswalk
        for _, cw in crosswalks.iterrows():
            cw_x1, cw_y1, cw_x2, cw_y2 = cw['xmin'], cw['ymin'], cw['xmax'], cw['ymax']
            # Nếu biển số ở trên crosswalk
            if py1 < cw_y2 and py2 > cw_y1 and abs(plate_center_x - (cw_x1 + cw_x2)/2) < 100:
                violations.append("Đè vạch dừng")
        
        # Phát hiện đậu sai nơi quy định (parking meter)
        parking_meters = df[df['name'] == 'parking meter']
        for _, pm in parking_meters.iterrows():
            pm_x1, pm_y1, pm_x2, pm_y2 = pm['xmin'], pm['ymin'], pm['xmax'], pm['ymax']
            if abs(pm_x1 - px1) < 150 and abs(pm_y1 - py1) < 150:
                violations.append("Đậu sai nơi quy định")
        
        # Phát hiện đè làn đường (dựa trên kích thước biển số hoặc vị trí)
        # Giả định nếu biển số lớn và ở giữa, có thể đè làn
        if px2 - px1 > 200 and plate_center_x > frame.shape[1] * 0.4 and plate_center_x < frame.shape[1] * 0.6:
            violations.append("Đè làn đường")
            
    except Exception as e:
        print(f"Lỗi phát hiện vi phạm: {e}")
    
    return violations

def process_frame(frame):
    """Xử lý frame cho realtime"""
    try:
        # Phát hiện biển số
        results = model_detect(frame)
        df = results.pandas().xyxy[0]
        valid = df[df['confidence'] > 0.6]

        if len(valid) > 0:
            row = valid.iloc[0]
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Phát hiện vi phạm
            violations = detect_violation(frame, (xmin, ymin, xmax, ymax))
            
            # OCR
            crop = frame[ymin:ymax, xmin:xmax]
            text = sap_xep_bien_so(model_ocr(crop, verbose=False, conf=0.5), ymax-ymin, xmax-xmin)
            
            # Xác định màu sắc
            if not text:
                color = (0, 0, 255)  # Đỏ
            elif violations:
                color = (0, 255, 255)  # Vàng
            else:
                color = (0, 255, 0)  # Xanh
            
            # Vẽ bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Cập nhật status
            global current_status
            if not text:
                current_status["plate"] = "Không rõ"
                current_status["status"] = "Không rõ"
                current_status["color"] = "danger"
            elif violations:
                current_status["plate"] = text
                current_status["status"] = f"Vi phạm: {', '.join(violations)}"
                current_status["color"] = "warning"
            else:
                current_status["plate"] = text
                current_status["status"] = "Không vi phạm"
                current_status["color"] = "success"
            current_status["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
    
    except Exception as e:
        print(f"Lỗi xử lý frame: {e}")
    
    return frame

def gen_frames():
    """Generator cho video feed realtime"""
    global camera, is_running
    if camera is None:
        camera = cv2.VideoCapture(0)
    while is_running:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    """Xử lý ảnh tĩnh"""
    frame = cv2.imread(image_path)
    if frame is None:
        return None, "Không thể đọc ảnh", "Lỗi", "danger", None, "Không rõ"
    info_text = ""
    try:
        # Detect
        results = model_detect(frame)
        df = results.pandas().xyxy[0]
        valid = df[df['confidence'] > 0.6]

        if len(valid) > 0:
            row = valid.iloc[0]
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Phân loại phương tiện dựa trên kích thước biển số
            plate_width = xmax - xmin
            if plate_width > 300:
                vehicle_type = "Xe bus"
            elif plate_width > 150:
                vehicle_type = "Ô tô"
            else:
                vehicle_type = "Xe máy"
            
            # Vẽ khung
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # OCR
            crop = frame[ymin:ymax, xmin:xmax]
            res_char = model_ocr(crop, verbose=False, conf=0.5)
            text, avg_conf = sap_xep_bien_so(res_char, ymax-ymin, xmax-xmin)

            # Phát hiện vi phạm
            violations = detect_violation(frame, (xmin, ymin, xmax, ymax))

            # Đánh giá độ rõ của biển số
            if avg_conf < 0.7:
                clarity = "Mờ"
                border_color = "danger"  # Đỏ
                info_text = f"{vehicle_type} - {text} (Mờ)"
            else:
                clarity = "Rõ"
                if violations:
                    border_color = "warning"  # Cam vàng
                    info_text = f"{vehicle_type} - {text} - Vi phạm: {', '.join(violations)}"
                else:
                    border_color = "success"  # Xanh lá
                    info_text = f"{vehicle_type} - {text}"

            # Vẽ khung với màu tương ứng
            if border_color == "success":
                color = (0, 255, 0)  # Xanh
            elif border_color == "warning":
                color = (0, 255, 255)  # Vàng
            else:
                color = (0, 0, 255)  # Đỏ
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Ghi log nếu có vi phạm
            if violations:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_to_csv(clean, "PHÁT HIỆN VI PHẠM", info_text, f"Ảnh: {os.path.basename(image_path)}")
            
            return frame, text, info_text, border_color, crop, clarity
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        # Re-raise so outer route can catch and display full traceback
        raise RuntimeError(tb)

def process_video(video_path):
    """Xử lý video: phát hiện biển số, đánh giá độ rõ, phân loại, vẽ bounding box và lưu video đã xử lý."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Không thể mở video", None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo đường dẫn video đã xử lý
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    processed_filename = f"processed_{base_name}.mp4"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

    plates_found = []
    seen = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_copy = frame.copy()  # Sao chép frame để vẽ

        if frame_count % 10 == 0:  # Xử lý mỗi 10 frame để tăng tốc và trực quan hơn
            try:
                results = model_detect(frame)
                df = results.pandas().xyxy[0]
                valid = df[df['confidence'] > 0.6]

                if len(valid) > 0:
                    row = valid.iloc[0]
                    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                    # Phân loại phương tiện
                    plate_width = xmax - xmin
                    if plate_width > 300:
                        vehicle_type = "Xe bus"
                    elif plate_width > 150:
                        vehicle_type = "Ô tô"
                    else:
                        vehicle_type = "Xe máy"

                    crop = frame[ymin:ymax, xmin:xmax]
                    res_char = model_ocr(crop, verbose=False, conf=0.5)
                    text, avg_conf = sap_xep_bien_so(res_char, ymax-ymin, xmax-xmin)

                    if not text:
                        out.write(frame_copy)
                        continue

                    clean = text.replace('-', '').replace('.', '').replace(' ', '').upper()
                    if clean in seen:
                        out.write(frame_copy)
                        continue
                    seen.add(clean)

                    # Phát hiện vi phạm
                    violations = detect_violation(frame, (xmin, ymin, xmax, ymax))

                    # Đánh giá độ rõ
                    clarity = "Mờ" if avg_conf < 0.7 else "Rõ"

                    # Xác định màu bounding box
                    if clarity == "Mờ":
                        color = (0, 0, 255)  # Đỏ
                        status_color = "danger"
                        info_text = f"{vehicle_type} - {text} (Mờ)"
                    else:
                        if violations:
                            color = (0, 165, 255)  # Cam
                            status_color = "warning"
                            info_text = f"{vehicle_type} - {text} - Vi phạm: {', '.join(violations)}"
                        else:
                            color = (0, 255, 0)  # Xanh lá
                            status_color = "success"
                            info_text = f"{vehicle_type} - {text}"

                    # Vẽ bounding box
                    cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), color, 3)
                    # Vẽ text
                    cv2.putText(frame_copy, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Lưu crop
                    crop_name = f"crop_{base_name}_{clean}_{frame_count}.jpg"
                    crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_name)
                    cv2.imwrite(crop_path, crop)
                    crop_url = url_for('static', filename='uploads/' + crop_name)

                    # Ghi log nếu có vi phạm
                    if violations:
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_to_csv(clean, "PHÁT HIỆN VI PHẠM", info_text, f"Video: {os.path.basename(video_path)} - Frame {frame_count}")

                    plates_found.append({
                        'plate': text,
                        'clean': clean,
                        'info': info_text,
                        'color': status_color,
                        'frame': frame_count,
                        'timestamp': round(frame_count / fps, 2),
                        'clarity': clarity,
                        'crop_url': crop_url,
                        'vehicle_type': vehicle_type
                    })
            except Exception:
                pass

        out.write(frame_copy)

    cap.release()
    out.release()

    # Ghi log cho video
    for plate_info in plates_found:
        log_to_csv(plate_info['clean'], "PHÁT HIỆN VI PHẠM VIDEO", plate_info['info'], f"Video: {os.path.basename(video_path)}, Frame: {plate_info['frame']}, Độ rõ: {plate_info['clarity']}")

    processed_video_url = url_for('static', filename='uploads/' + processed_filename)
    return plates_found, f"Tìm thấy {len(plates_found)} biển số trong video", processed_video_url

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
            text, avg_conf = sap_xep_bien_so(res_char, ymax-ymin, xmax-xmin)

            if len(text) > 4:
                # Logic kiểm tra cư dân
                clean = text.replace('-', '').replace('.', '').replace(' ', '').upper()
                info_text = "KHÁCH VÃNG LAI"
                status_color = "dark"  # Thay warning thành dark
                conf = avg_conf  # Thêm conf
                
                # Check DB
                for db_plate, info in db_cudan.items():
                    if difflib.SequenceMatcher(None, clean, db_plate).ratio() > 0.8:
                        info_text = f"CƯ DÂN: {info['name']}"
                        status_color = "secondary"  # Thay success thành secondary
                        clean = db_plate
                        break
                
                # Cập nhật biến toàn cục để Web đọc được
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                if current_status["plate"] != clean: # Chỉ log khi biển số thay đổi
                    current_status = {
                        "plate": clean,
                        "status": info_text,
                        "color": status_color,
                        "timestamp": now_str,
                        "conf": conf
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

# API: Ghi nhận biển số thủ công khi nhấn Space
@app.route('/api/manual_capture', methods=['POST'])
def manual_capture():
    plate = current_status['plate']
    if plate == '---':
        return jsonify({"success": False, "message": "Không có biển số để ghi nhận!"})
    
    # Capture frame hiện tại và process để get crop
    if camera is not None:
        success, frame = camera.read()
        if success:
            # Process frame để get crop
            try:
                results = model_detect(frame)
                df = results.pandas().xyxy[0]
                valid = df[df['confidence'] > 0.6]
                if len(valid) > 0:
                    row = valid.iloc[0]
                    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    crop = frame[ymin:ymax, xmin:xmax]
                    # Lưu crop
                    crop_name = f"manual_{plate}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_name)
                    cv2.imwrite(crop_path, crop)
                    crop_url = url_for('static', filename='uploads/' + crop_name)
                else:
                    crop_url = None
            except:
                crop_url = None
        else:
            crop_url = None
    else:
        crop_url = None
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_csv(plate, "GHI NHẬN THỦ CÔNG", "Xe", "Nhấn Space")
    return jsonify({"success": True, "message": f"Đã ghi nhận biển số {plate}!", "crop_url": crop_url})
@app.route('/api/manual_open', methods=['POST'])
def manual_open():
    # Code gửi tín hiệu xuống Arduino ở đây
    print(">>> MỞ BARIE THỦ CÔNG")
    return jsonify({"success": True, "message": "Đã mở cổng!"})

# --- ROUTES MỚI ---
@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Xử lý ảnh
            processed_frame, plate_text, status, color, crop, clarity = process_image(filepath)

            if processed_frame is not None:
                # Lưu ảnh đã xử lý
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
                cv2.imwrite(result_path, processed_frame)

                # Lưu ảnh biển số cropped nếu có
                if crop is not None:
                    crop_path = os.path.join(app.config['UPLOAD_FOLDER'], 'crop_' + filename)
                    cv2.imwrite(crop_path, crop)
                    crop_url = url_for('static', filename='uploads/crop_' + filename)
                else:
                    crop_url = None

                return render_template('upload_result.html', 
                                     plate=plate_text, 
                                     status=status, 
                                     color=color, 
                                     clarity=clarity,
                                     image_url=url_for('static', filename='uploads/result_' + filename),
                                     crop_url=crop_url)
            else:
                return render_template('upload_result.html', error=plate_text)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            return render_template('upload_result.html', error=f"Lỗi nội bộ: {str(e)}\n{tb}")
    
    return render_template('upload_image.html')

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Xử lý video
        plates, message, processed_video_url = process_video(filepath)

        return render_template('upload_video_result.html',
                     plates=plates,
                     message=message,
                     video_url=url_for('static', filename='uploads/' + filename),
                     processed_video_url=processed_video_url)
    
    return render_template('upload_video_v2.html')

@app.route('/history')
def history():
    if os.path.exists(FILE_LICH_SU):
        df = pd.read_csv(FILE_LICH_SU)
        data = df.to_dict(orient='records')
        return render_template('history.html', history=data)
    return render_template('history.html', history=[])

@app.route('/stats')
def stats():
    if os.path.exists(FILE_LICH_SU):
        df = pd.read_csv(FILE_LICH_SU)
        total_detections = len(df)
        unique_plates = df['BienSo'].nunique()
        resident_count = len(df[df['DoiTuong'].str.contains('MIỄN')])
        guest_count = len(df[df['DoiTuong'].str.contains('VI PHẠM')])
        
        return render_template('stats.html', 
                             total=total_detections, 
                             unique=unique_plates, 
                             resident=resident_count, 
                             guest=guest_count)
    return render_template('stats.html', total=0, unique=0, resident=0, guest=0)

@app.route('/export_excel')
def export_excel():
    if os.path.exists(FILE_LICH_SU):
        df = pd.read_csv(FILE_LICH_SU)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='LichSu', index=False)
        output.seek(0)
        return send_file(output, 
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name='lich_su_ra_vao.xlsx')
    return "Không có dữ liệu để xuất"

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_realtime')
def start_realtime():
    global is_running
    is_running = True
    return redirect(url_for('realtime'))

@app.route('/stop_realtime')
def stop_realtime():
    global is_running, camera
    is_running = False
    if camera is not None:
        camera.release()
        camera = None
    return redirect(url_for('index'))

@app.route('/api/status')
def api_status():
    return jsonify(current_status)

# === API CHO UPLOAD IMAGE ===
@app.route('/api/detect_image', methods=['POST'])
def api_detect_image():
    """API để xử lý ảnh upload và trả về kết quả JSON"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Không có file được tải'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'File không hợp lệ'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Xử lý ảnh
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({'success': False, 'message': 'Không thể đọc ảnh'}), 400
        
        # Detect biển số
        results = model_detect(frame)
        df = results.pandas().xyxy[0]
        valid = df[df['confidence'] > 0.6]
        
        if len(valid) == 0:
            # Không tìm thấy biển số
            return jsonify({
                'success': True,
                'plate': '',
                'clarity': 'Mờ',
                'vehicle_type': 'Không xác định',
                'status': 'Không nhận diện',
                'violations': [],
                'info_text': 'Hình ảnh mờ hoặc không thể nhận diện biển số',
                'image_url': url_for('static', filename='uploads/' + filename),
                'crop_url': None
            })
        
        row = valid.iloc[0]
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Phân loại phương tiện
        plate_width = xmax - xmin
        if plate_width > 300:
            vehicle_type = "Xe bus"
        elif plate_width > 150:
            vehicle_type = "Ô tô"
        else:
            vehicle_type = "Xe máy"
        
        # Crop biển số
        crop = frame[ymin:ymax, xmin:xmax]
        
        # OCR
        res_char = model_ocr(crop, verbose=False, conf=0.5)
        text, avg_conf = sap_xep_bien_so(res_char, ymax-ymin, xmax-xmin)
        
        # Phát hiện vi phạm
        violations = detect_violation(frame, (xmin, ymin, xmax, ymax))
        
        # Đánh giá độ rõ
        clarity = "Rõ" if avg_conf >= 0.7 else "Mờ"
        
        # Lưu ảnh biển số cắt
        crop_filename = f"crop_{os.path.splitext(filename)[0]}.jpg"
        crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
        cv2.imwrite(crop_path, crop)
        crop_url = url_for('static', filename='uploads/' + crop_filename)
        
        # Vẽ khung trên ảnh gốc
        if clarity == "Rõ":
            if violations:
                color = (0, 165, 255)  # Vàng cam (BGR)
            else:
                color = (0, 255, 0)  # Xanh (BGR)
        else:
            color = (0, 0, 255)  # Đỏ (BGR)
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        
        # Lưu ảnh đã xử lý
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, frame)
        image_url = url_for('static', filename='uploads/' + result_filename)
        
        # Xây dựng thông tin chi tiết
        if not text or clarity == "Mờ":
            status = "Hình ảnh mờ"
            info_text = f"{vehicle_type} - Hình ảnh mờ, không rõ"
            plate_display = ""  # Không hiển thị biển số nếu mờ
        else:
            plate_display = text
            if violations:
                status = f"Vi phạm: {', '.join(violations)}"
                info_text = f"{vehicle_type} - {text} - {status}"
            else:
                status = "Không vi phạm"
                info_text = f"{vehicle_type} - {text} - Bình thường"
        
        # Ghi log nếu có vi phạm
        if violations and text:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_to_csv(text, "PHÁT HIỆN VI PHẠM", info_text, f"Ảnh: {filename}")
        
        return jsonify({
            'success': True,
            'plate': plate_display,
            'clarity': clarity,
            'vehicle_type': vehicle_type,
            'status': status,
            'violations': violations,
            'info_text': info_text,
            'image_url': image_url,
            'crop_url': crop_url
        })
        
    except Exception as e:
        print(f"Lỗi API detect_image: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Lỗi nội bộ: {str(e)}'}), 500


@app.route('/api/process_video', methods=['POST'])
def api_process_video():
    """API để xử lý video upload và trả về kết quả JSON"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Không có file được tải'}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'File không hợp lệ'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        plates, message = process_video(filepath)
        return jsonify({'success': True, 'message': message, 'plates': plates, 'video_url': url_for('static', filename='uploads/' + filename)})
    except Exception as e:
        print(f"Lỗi API process_video: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/save_detection', methods=['POST'])
def api_save_detection():
    """API để lưu kết quả nhận diện vào CSV"""
    try:
        data = request.get_json()
        plate = data.get('plate', '')
        status = data.get('status', '')
        vehicle_type = data.get('vehicle_type', '')
        violations = data.get('violations', [])
        
        if plate:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_text = f"{vehicle_type} - {status}"
            log_to_csv(plate, "NHẬN DIỆN THỦ CÔNG", info_text, f"Từ upload ảnh: {', '.join(violations)}")
            return jsonify({'success': True, 'message': 'Đã lưu kết quả'})
        
        return jsonify({'success': False, 'message': 'Không có biển số để lưu'})
    
    except Exception as e:
        print(f"Lỗi API save_detection: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)