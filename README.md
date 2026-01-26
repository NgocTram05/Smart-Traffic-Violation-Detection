Bước 1: Tải mã nguồn về máy
Bash
git clone https://github.com/NgocTram05/Smart-Parking.git
cd Smart-Parking
Bước 2: Tạo môi trường ảo
Giúp tránh xung đột thư viện với các dự án khác.

Bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# MacOS/Linux
python3 -m venv .venv
source .venv/bin/activate
Bước 3: Cài đặt thư viện cần thiết
Bash
pip install -r requirements.txt

Chạy chương trình
Mở Terminal tại thư mục dự án và chạy lệnh:

Bash
python main.py