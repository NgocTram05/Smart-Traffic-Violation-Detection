import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
import pathlib
import sys
from ultralytics import YOLO

# --- CẤU HÌNH ---
# Sửa lại đường dẫn model nếu cần
PATH_DETECT = 'model/LP_detector.pt'  # Model phát hiện vùng biển số (YOLOv5)
PATH_OCR = 'model/model_nhandien_kytu.pt' # Model đọc chữ (YOLOv8)

class SimpleLPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SIMPLE LPR - NHẬN DIỆN BIỂN SỐ XE")
        self.root.geometry("900x600")
        self.root.configure(bg="white")

        # --- 1. LOAD MODEL ---
        self.load_models()

        # --- 2. GIAO DIỆN ---
        # Tiêu đề
        title_label = tk.Label(root, text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE", font=("Arial", 18, "bold"), bg="white", fg="black")
        title_label.pack(pady=(20, 10))

        # Khung Camera
        camera_frame = tk.Frame(root, bg="white", relief="ridge", bd=2)
        camera_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        self.lbl_video = tk.Label(camera_frame, bg="white", text="Đang tải camera...", font=("Arial", 12))
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # Khung Kết Quả
        result_frame = tk.Frame(root, bg="lightgray", relief="ridge", bd=2, padx=10, pady=10)
        result_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(result_frame, text="BIỂN SỐ ĐỌC ĐƯỢC:", font=("Arial", 14, "bold"), bg="lightgray", fg="black").pack(anchor="w")
        self.lbl_result = tk.Label(result_frame, text="---", font=("Arial", 28, "bold"), bg="lightgray", fg="blue")
        self.lbl_result.pack(pady=(5, 10))

        # Ảnh Biển Số Cắt Ra
        crop_frame = tk.Frame(result_frame, bg="white", relief="sunken", bd=1)
        crop_frame.pack(fill=tk.X, pady=5)
        self.lbl_crop = tk.Label(crop_frame, bg="white", text="Chưa có ảnh biển số")
        self.lbl_crop.pack(pady=5)

        # Hướng dẫn
        instruction_label = tk.Label(root, text="💡 Bấm phím SPACE để chụp và đọc biển số", font=("Arial", 11), bg="white", fg="gray")
        instruction_label.pack(side=tk.BOTTOM, pady=(10, 20))

        # --- 3. KHỞI ĐỘNG CAMERA ---
        # Thử camera trước, nếu không được thì dùng video mẫu
        self.cap = cv2.VideoCapture(0) # Số 0 là Webcam
        if not self.cap.isOpened():
            print("⚠️ Không tìm thấy camera, thử dùng video mẫu...")
            # Có thể thêm video mẫu ở đây nếu có
            self.lbl_video.configure(text="Không có camera hoặc video", font=("Arial", 12))
            return

        # Bắt sự kiện phím SPACE
        self.root.bind('<space>', self.trigger_detection)

    def load_models(self):
        print("⏳ Đang tải model...")
        try:
            # Add yolov5 to path for model loading
            sys.path.append('yolov5')
            
            # Fix pathlib for Windows
            pathlib.PosixPath = pathlib.WindowsPath
            
            # Load Model Detect (YOLOv5 Custom)
            self.model_detect = YOLO(PATH_DETECT)
            
            # Load Model OCR (YOLOv8)
            self.model_ocr = YOLO(PATH_OCR)
            
            print("✅ Model đã sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            messagebox.showerror("Lỗi", f"Không tìm thấy model!\n{e}")

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            # Resize hiển thị
            frame_resized = cv2.resize(frame, (640, 360))
            # Chuyển màu BGR -> RGB để hiện lên Tkinter
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
        
        self.root.after(10, self.update_camera)

    def trigger_detection(self, event):
        """Khi bấm SPACE thì gọi hàm này"""
        frame = self.current_frame.copy()
        
        # 1. Phát hiện biển số (Detect)
        results = self.model_detect(frame)
        df = results.pandas().xyxy[0]  # Lấy kết quả dưới dạng Pandas DataFrame
        
        # Lấy biển số có độ tin cậy cao nhất
        valid_plates = df[df['confidence'] > 0.5]
        
        if len(valid_plates) > 0:
            # Lấy tọa độ biển số đầu tiên
            plate = valid_plates.iloc[0]
            xmin, ymin, xmax, ymax = int(plate['xmin']), int(plate['ymin']), int(plate['xmax']), int(plate['ymax'])
            
            # Cắt ảnh biển số
            crop_img = frame[ymin:ymax, xmin:xmax]
            
            # 2. Đọc ký tự (OCR)
            text = self.ocr_process(crop_img)
            
            # 3. Hiển thị kết quả
            self.lbl_result.configure(text=text, fg="#2ecc71") # Màu xanh lá
            
            # Hiển thị ảnh crop lên giao diện
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # Resize ảnh crop cho dễ nhìn (cao 80px)
            h, w, _ = crop_rgb.shape
            scale = 80 / h
            crop_resized = cv2.resize(crop_rgb, (int(w * scale), 80))
            img_crop_tk = ImageTk.PhotoImage(image=Image.fromarray(crop_resized))
            self.lbl_crop.configure(image=img_crop_tk)
            self.lbl_crop.image = img_crop_tk
            
            # Vẽ khung chữ nhật lên màn hình video (nháy 1 cái cho đẹp)
            print(f"✅ Đọc được: {text}")

        else:
            self.lbl_result.configure(text="Không tìm thấy!", fg="#e74c3c") # Màu đỏ
            self.lbl_crop.configure(image='')
            print("⚠️ Không thấy biển số nào.")

    def ocr_process(self, crop_img):
        """Hàm đọc ký tự từ ảnh cắt với cải thiện độ chính xác"""
        # Tiền xử lý ảnh để cải thiện nhận diện
        crop_img = self.preprocess_image(crop_img)
        
        results = self.model_ocr(crop_img, verbose=False)
        
        chars = []
        for result in results:
            for box in result.boxes:
                # Lấy confidence
                conf = float(box.conf[0])
                
                # Chỉ chấp nhận ký tự có confidence > 0.6
                if conf < 0.6:
                    continue
                
                # Lấy class ID
                cls_id = int(box.cls[0])
                char_str = self.model_ocr.names[cls_id]
                
                # Validate ký tự hợp lệ cho biển số Việt Nam
                if not self.is_valid_plate_char(char_str):
                    continue
                
                # Lấy tọa độ
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                chars.append({'char': char_str, 'cx': cx, 'cy': cy, 'conf': conf})
        
        if not chars:
            return "Không đọc được"

        # Sắp xếp và xử lý như cũ
        h, w, _ = crop_img.shape
        if h / w > 0.4: 
            avg_cy = sum(c['cy'] for c in chars) / len(chars)
            top_row = sorted([c for c in chars if c['cy'] < avg_cy], key=lambda x: x['cx'])
            bot_row = sorted([c for c in chars if c['cy'] >= avg_cy], key=lambda x: x['cx'])
            
            text_top = "".join([c['char'] for c in top_row])
            text_bot = "".join([c['char'] for c in bot_row])
            return f"{text_top}-{text_bot}"
        else:
            chars.sort(key=lambda x: x['cx'])
            return "".join([c['char'] for c in chars])

    def preprocess_image(self, img):
        """Tiền xử lý ảnh để cải thiện nhận diện"""
        # Chuyển sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tăng độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Làm sắc nét
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Chuyển lại thành 3 kênh để YOLO xử lý
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def is_valid_plate_char(self, char):
        """Kiểm tra ký tự có hợp lệ cho biển số Việt Nam không"""
        # Biển số Việt Nam gồm: chữ cái A-Z (trừ I, O, Q), số 0-9
        valid_chars = "0123456789ABCDEFGHKLMNPSTUVXYZ"
        return char in valid_chars

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleLPRApp(root)
    root.mainloop()