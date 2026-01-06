import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import pandas as pd
import datetime
import os
import difflib
import numpy as np
from ultralytics import YOLO

FILE_CU_DAN = 'cu_dan.csv'
FILE_LICH_SU = 'lich_su_ra_vao.csv'

PATH_DETECT = 'model/LP_detector.pt'   
PATH_OCR = 'model/model_nhandien_kytu.pt'

XE_TRONG_BAI = {} 

class ParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG AN NINH CHUNG CƯ CAO CẤP (V3.1 FINAL)")
        self.root.geometry("1280x760")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook.Tab", font=('Arial', 12, 'bold'), padding=[20, 10])

        # --- LOAD DỮ LIỆU & MODEL ---
        self.load_data()
        self.load_models() # Hàm này sẽ load model dựa trên đường dẫn ở trên

        # --- TẠO TAB ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_dashboard = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_dashboard, text="📺 GIÁM SÁT TRỰC TIẾP")

        self.tab_history = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_history, text="📜 LỊCH SỬ RA VÀO")

        # --- GIAO DIỆN ---
        self.setup_dashboard_ui()
        self.setup_history_ui()
        
        # --- CAMERA ---
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.update_camera()

        self.root.bind('<space>', self.simulate_sensor_trigger)

    def load_data(self):
        self.db_cudan = {}
        try:
            df = pd.read_csv(FILE_CU_DAN)
            for _, row in df.iterrows():
                clean_plate = str(row['bien_so']).replace('.', '').replace('-', '').replace(' ', '').upper()
                self.db_cudan[clean_plate] = {'name': row['ten_chu_xe'], 'room': row['so_phong']}
            print(f"✅ Đã tải {len(self.db_cudan)} cư dân.")
        except:
            print("⚠️ Chưa có file cu_dan.csv (Chế độ chạy không dữ liệu)")

    def load_models(self):
        print(f"⏳ Đang tải model từ: {PATH_DETECT}")
        try:
            # Fix lỗi Path Windows khi load model
            import pathlib
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            
            # Load model Detect
            self.model_detect = torch.hub.load('yolov5', 'custom', path=PATH_DETECT, source='local', force_reload=True)
            
            # Load model OCR
            self.model_ocr = YOLO(PATH_OCR)
            print("✅ Model AI Sẵn sàng!")
        except Exception as e:
            messagebox.showerror("Lỗi Model", f"Không tìm thấy file Model!\nHãy kiểm tra lại thư mục 'model'.\nChi tiết lỗi: {e}")
            print(f"❌ LỖI: {e}")

    # ... (Các phần giao diện giữ nguyên như cũ) ...
    def setup_dashboard_ui(self):
        top_frame = tk.Frame(self.tab_dashboard, bg="#f0f0f0")
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        cam_frame = tk.LabelFrame(top_frame, text="CAMERA CỔNG CHÍNH", font=("Arial", 10, "bold"), bg="black", fg="white")
        cam_frame.pack()
        self.lbl_video = tk.Label(cam_frame, bg="black")
        self.lbl_video.pack()

        info_container = tk.Frame(self.tab_dashboard, bg="#f0f0f0")
        info_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Cột Trái (Vào)
        frame_in = tk.LabelFrame(info_container, text="⬇️ LỐI VÀO", font=("Arial", 12, "bold"), bg="white", fg="#2E7D32")
        frame_in.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.lbl_plate_in_img = tk.Label(frame_in, text="[Ảnh Xe Vào]", bg="#eee")
        self.lbl_plate_in_img.pack(pady=5, fill=tk.X, padx=10)
        self.lbl_plate_in_text = tk.Label(frame_in, text="---", font=("Arial", 24, "bold"), fg="#2E7D32", bg="white")
        self.lbl_plate_in_text.pack(pady=2)
        self.txt_info_in = tk.Label(frame_in, text="Đang chờ xe...", font=("Arial", 11), bg="white", justify="center")
        self.txt_info_in.pack(pady=5)

        # Cột Phải (Ra)
        frame_out = tk.LabelFrame(info_container, text="⬆️ LỐI RA", font=("Arial", 12, "bold"), bg="white", fg="#C62828")
        frame_out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.lbl_plate_out_img = tk.Label(frame_out, text="[Ảnh Xe Ra]", bg="#eee")
        self.lbl_plate_out_img.pack(pady=5, fill=tk.X, padx=10)
        self.lbl_plate_out_text = tk.Label(frame_out, text="---", font=("Arial", 24, "bold"), fg="#C62828", bg="white")
        self.lbl_plate_out_text.pack(pady=2)
        self.txt_info_out = tk.Label(frame_out, text="Đang chờ xe...", font=("Arial", 11), bg="white", justify="center")
        self.txt_info_out.pack(pady=5)

        # Footer
        control_frame = tk.Frame(self.tab_dashboard, bg="#ddd", height=50)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(control_frame, text="⚠️ MỞ CỔNG KHẨN CẤP (BARIE)", bg="orange", fg="white", font=("Arial", 11, "bold"), 
                  command=self.manual_open).pack(pady=10, fill=tk.X, padx=100)

    def setup_history_ui(self):
        toolbar = tk.Frame(self.tab_history, bg="#ddd")
        toolbar.pack(fill=tk.X)
        tk.Button(toolbar, text="🔄 Làm mới danh sách", command=self.load_history_to_table).pack(side=tk.LEFT, padx=10, pady=5)

        columns = ("Time", "Plate", "Action", "Person", "Note")
        self.tree = ttk.Treeview(self.tab_history, columns=columns, show="headings")
        self.tree.heading("Time", text="Thời Gian")
        self.tree.heading("Plate", text="Biển Số")
        self.tree.heading("Action", text="Hành Động")
        self.tree.heading("Person", text="Đối Tượng")
        self.tree.heading("Note", text="Trạng Thái")
        self.tree.column("Time", width=120, anchor="center")
        self.tree.column("Plate", width=120, anchor="center")
        self.tree.column("Action", width=80, anchor="center")
        self.tree.column("Person", width=250, anchor="w")
        self.tree.column("Note", width=150, anchor="center")
        
        scrollbar = ttk.Scrollbar(self.tab_history, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.load_history_to_table()

    # ... (Logic xử lý AI & Update Camera) ...
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            img = cv2.cvtColor(cv2.resize(frame, (480, 280)), cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
        if self.is_running:
            self.root.after(10, self.update_camera)

    def simulate_sensor_trigger(self, event):
        plate_text, plate_img = self.ai_process(self.current_frame)
        if plate_text:
            self.process_parking_logic(plate_text, plate_img)
        else:
            print("⚠️ Không tìm thấy biển số -> Reset màn hình")
            self.lbl_plate_in_text.configure(text="---", fg="black")
            self.lbl_plate_out_text.configure(text="---", fg="black")
            self.txt_info_in.configure(text="⚠️ KHÔNG THẤY BIỂN SỐ", fg="#FF9800")
            self.txt_info_out.configure(text="⚠️ Vui lòng thử lại", fg="#FF9800")
            
            blank_image = np.zeros((60, 200, 3), np.uint8) 
            blank_image.fill(200)
            img_blank = ImageTk.PhotoImage(image=Image.fromarray(blank_image))
            self.lbl_plate_in_img.configure(image=img_blank)
            self.lbl_plate_in_img.image = img_blank
            self.lbl_plate_out_img.configure(image=img_blank)
            self.lbl_plate_out_img.image = img_blank

    def process_parking_logic(self, plate_raw, plate_img):
        plate_clean = plate_raw.replace('-', '').replace('.', '').replace(' ', '').upper()
        is_resident = False
        person_info = "KHÁCH VÃNG LAI"
        for db_plate, info in self.db_cudan.items():
            if difflib.SequenceMatcher(None, plate_clean, db_plate).ratio() > 0.8:
                is_resident = True
                person_info = f"CƯ DÂN: {info['name']}\n({info['room']})"
                plate_clean = db_plate 
                break

        now = datetime.datetime.now().strftime("%H:%M:%S")
        img_crop_gui = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)).resize((200, 60)))

        if plate_clean in XE_TRONG_BAI:
            gio_vao = XE_TRONG_BAI.pop(plate_clean)
            self.lbl_plate_out_text.configure(text=plate_raw)
            self.lbl_plate_out_img.configure(image=img_crop_gui)
            self.lbl_plate_out_img.image = img_crop_gui 
            msg = f"{person_info}\nGiờ vào: {gio_vao}\nGiờ ra: {now}\n[ĐÃ MỞ CỔNG]"
            self.txt_info_out.configure(text=msg, fg="blue")
            self.save_log(now, plate_raw, "RA", person_info, "Hoàn tất")
        else:
            XE_TRONG_BAI[plate_clean] = now
            self.lbl_plate_in_text.configure(text=plate_raw)
            self.lbl_plate_in_img.configure(image=img_crop_gui)
            self.lbl_plate_in_img.image = img_crop_gui 
            msg = f"{person_info}\nGiờ vào: {now}\n[ĐƯỢC PHÉP VÀO]"
            self.txt_info_in.configure(text=msg, fg="green")
            self.save_log(now, plate_raw, "VÀO", person_info, "Đang gửi")

    def save_log(self, time, plate, action, person, note):
        if not os.path.exists(FILE_LICH_SU):
             pd.DataFrame(columns=['ThoiGian', 'BienSo', 'HanhDong', 'DoiTuong', 'GhiChu']).to_csv(FILE_LICH_SU, index=False)
        new_row = {'ThoiGian': time, 'BienSo': plate, 'HanhDong': action, 'DoiTuong': person.replace('\n', ' '), 'GhiChu': note}
        df = pd.DataFrame([new_row])
        df.to_csv(FILE_LICH_SU, mode='a', header=False, index=False)
        self.tree.insert("", 0, values=(time, plate, action, person.replace('\n', ' '), note))

    def load_history_to_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if os.path.exists(FILE_LICH_SU):
            try:
                df = pd.read_csv(FILE_LICH_SU)
                for _, row in df.iloc[::-1].iterrows(): 
                    self.tree.insert("", "end", values=(row['ThoiGian'], row['BienSo'], row['HanhDong'], row['DoiTuong'], row['GhiChu']))
            except: pass

    def manual_open(self):
        messagebox.showinfo("Thông báo", "Đã kích hoạt mở cổng khẩn cấp!")

    def ai_process(self, frame):
        results = self.model_detect(frame)
        df_results = results.pandas().xyxy[0]
        df_valid = df_results[df_results['confidence'] > 0.6]
        plates = df_valid.values.tolist()
        
        if len(plates) == 0: return None, None
        plate = plates[0] 
        xmin, ymin, xmax, ymax = int(plate[0]), int(plate[1]), int(plate[2]), int(plate[3])
        crop = frame[ymin:ymax, xmin:xmax]
        
        results_char = self.model_ocr(crop, verbose=False, conf=0.5)
        text = self.sap_xep_bien_so(results_char, ymax-ymin, xmax-xmin)
        if len(text) < 4: return None, None
        return text, crop

    def sap_xep_bien_so(self, results, height, width):
        chars = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                char_name = self.model_ocr.names[cls_id]
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkingApp(root)
    root.mainloop()