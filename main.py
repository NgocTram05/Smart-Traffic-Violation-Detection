import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
import pathlib
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
        self.root.configure(bg="#2c3e50")

        # --- 1. LOAD MODEL ---
        self.load_models()

        # --- 2. GIAO DIỆN ---
        # Tiêu đề
        tk.Label(root, text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ", font=("Arial", 20, "bold"), bg="#2c3e50", fg="#ecf0f1").pack(pady=10)

        # Khung Camera
        self.lbl_video = Label(root, bg="black")
        self.lbl_video.pack(pady=10)

        # Khung Kết Quả
        result_frame = tk.Frame(root, bg="#34495e", pady=10)
        result_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(result_frame, text="BIỂN SỐ ĐỌC ĐƯỢC:", font=("Arial", 14), bg="#34495e", fg="#bdc3c7").pack()
        self.lbl_result = tk.Label(result_frame, text="---", font=("Arial", 30, "bold"), bg="#34495e", fg="#f1c40f")
        self.lbl_result.pack()

        # Ảnh Biển Số Cắt Ra
        self.lbl_crop = Label(result_frame, bg="#34495e")
        self.lbl_crop.pack(pady=5)

        # Hướng dẫn
        tk.Label(root, text="👉 Bấm phím SPACE (Cách) để chụp và đọc biển số", font=("Arial", 12, "italic"), bg="#2c3e50", fg="#95a5a6").pack(side=tk.BOTTOM, pady=10)

        # --- 3. KHỞI ĐỘNG CAMERA ---
        self.cap = cv2.VideoCapture(0) # Số 0 là Webcam, hoặc thay bằng đường dẫn video
        self.update_camera()

        # Bắt sự kiện phím SPACE
        self.root.bind('<space>', self.trigger_detection)

    def load_models(self):
        print("⏳ Đang tải model...")
        try:
            # Fix lỗi Path Windows khi load model YOLOv5
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            
            # Load Model Detect (YOLOv5 Custom)
            self.model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH_DETECT, force_reload=True)
            
            # Load Model OCR (YOLOv8)
            self.model_ocr = YOLO(PATH_OCR)
            
            print("✅ Model đã sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            tk.messagebox.showerror("Lỗi", f"Không tìm thấy model!\n{e}")

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
        """Hàm đọc ký tự từ ảnh cắt"""
        results = self.model_ocr(crop_img, verbose=False)
        
        chars = []
        for result in results:
            for box in result.boxes:
                # Lấy class ID (0, 1, 2...) -> chuyển thành ký tự (A, B, 1, 2...)
                cls_id = int(box.cls[0])
                char_str = self.model_ocr.names[cls_id]
                
                # Lấy tọa độ để sắp xếp
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                chars.append({'char': char_str, 'cx': cx, 'cy': cy})
        
        if not chars:
            return "???"

        # Sắp xếp ký tự: 
        # Nếu biển vuông (2 dòng) -> Sắp theo Y trước, rồi X
        # Nếu biển dài (1 dòng) -> Sắp theo X
        
        h, w, _ = crop_img.shape
        # Logic đơn giản: Nếu chiều cao > 1/2 chiều rộng thì khả năng là biển vuông
        if h / w > 0.4: 
            # Chia 2 dòng dựa vào trung bình Y
            avg_cy = sum(c['cy'] for c in chars) / len(chars)
            top_row = sorted([c for c in chars if c['cy'] < avg_cy], key=lambda x: x['cx'])
            bot_row = sorted([c for c in chars if c['cy'] >= avg_cy], key=lambda x: x['cx'])
            
            text_top = "".join([c['char'] for c in top_row])
            text_bot = "".join([c['char'] for c in bot_row])
            return f"{text_top}-{text_bot}"
        else:
            # Biển dài -> Sắp xếp trái qua phải
            chars.sort(key=lambda x: x['cx'])
            return "".join([c['char'] for c in chars])

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleLPRApp(root)
    root.mainloop()