import cv2
import os
from glob import glob
from tqdm import tqdm  # Hiển thị thanh tiến trình

# Thư mục đầu vào và đầu ra
input_folder = "E:\\gearchuaxuly\\Anh1"
output_folder = "E:\\geardaxuly\\images1"

# Tạo thư mục nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Các định dạng ảnh hỗ trợ
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

# Tìm tất cả các ảnh có định dạng tương ứng
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(input_folder, ext)))

# Duyệt và resize ảnh, có thanh tiến trình
for img_path in tqdm(image_paths, desc="Đang xử lý ảnh"):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không đọc được ảnh: {img_path}")
            continue

        resized_img = cv2.resize(img, (640, 640))
        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_folder, img_name)

        cv2.imwrite(save_path, resized_img)
    except Exception as e:
        print(f"Lỗi với ảnh {img_path}: {e}")