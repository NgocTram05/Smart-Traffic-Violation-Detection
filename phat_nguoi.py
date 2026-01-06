import cv2
import torch
import numpy as np
import datetime
import os
import pathlib
from ultralytics import YOLO

VIDEO_PATH = 'traffic_video.mp4'
FOLDER_VI_PHAM = 'hinh_anh_vi_pham'

STOP_POLYGON = np.array([
    [350, 450],  # Góc trên Trái
    [950, 450],  # Góc trên Phải
    [1200, 720], # Góc dưới Phải
    [100, 720]   # Góc dưới Trái
], np.int32)

PATH_LP_DETECT = 'model/LP_detector.pt'
PATH_OCR = 'model/model_nhandien_kytu.pt'

if not os.path.exists(FOLDER_VI_PHAM): os.makedirs(FOLDER_VI_PHAM)

print("⏳ Đang tải Model...")
traffic_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
traffic_model.classes = [2, 3, 5, 7] 

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
lp_model = torch.hub.load('yolov5', 'custom', path=PATH_LP_DETECT, source='local', force_reload=True)
ocr_model = YOLO(PATH_OCR)
print("✅ Sẵn sàng!")

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

cap = cv2.VideoCapture(VIDEO_PATH)
is_red_light = False 
font = cv2.FONT_HERSHEY_SIMPLEX
known_plates = {} 

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        known_plates.clear()
        continue
    
    frame = cv2.resize(frame, (1280, 720))
    
    color_zone = (0, 0, 255) if is_red_light else (0, 255, 0)
    cv2.polylines(frame, [STOP_POLYGON], True, color_zone, 2)
    
    status_text = "DEN DO" if is_red_light else "DEN XANH"
    cv2.putText(frame, status_text, (1050, 70), font, 1.5, color_zone, 3)

    results = traffic_model(frame)
    df = results.pandas().xyxy[0]
    
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        check_point = (int((x1+x2)/2), y2) # Điểm bánh sau
        car_id = f"{x1//30}-{y1//30}"

        is_violation = is_red_light and is_in_stop_zone(check_point, STOP_POLYGON)

        if is_violation:
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
            
       

    cv2.imshow("He thong Phat Nguoi (Clean Mode)", frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == 27: break
    elif key == 32: 
        is_red_light = not is_red_light
        if not is_red_light: known_plates.clear()

cap.release()
cv2.destroyAllWindows()