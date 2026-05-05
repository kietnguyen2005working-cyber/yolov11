import cv2
from ultralytics import YOLO

import os

def main():
    # 1. Cấu hình đường dẫn 

    model_path = "best120.pt"
    video_path = "testvideo1.mp4" 
    
    # 2. Kiểm tra file 
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy file mô hình '{model_path}'")
        return
    if not os.path.exists(video_path):
        print(f"LỖI: Không tìm thấy file video '{video_path}'")
        return

    # 3. Load mô hình YOLO
    print("Đang khởi tạo mô hình...")
    model = YOLO(model_path)
    # THÊM DÒNG NÀY ĐỂ KIỂM TRA PHIÊN BẢN 
    print(f"--- Đang sử dụng phiên bản: {model.info()[0]} ---")
    # 4. Mở video
    cap = cv2.VideoCapture(video_path)
    
    # --- THÔNG SỐ GIỚI HẠN MÀN HÌNH ---
    MAX_DISPLAY_WIDTH = 640
    MAX_DISPLAY_HEIGHT = 600 

    print(" Nhấn 'q' tại cửa sổ video để thoát.")

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # 5. Chạy YOLO trên từng khung hình
            results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
            # 6. Vẽ khung nhận diện lên ảnh
            annotated_frame = results[0].plot()
            h_orig, w_orig = annotated_frame.shape[:2]  
            scaling_factor = min(MAX_DISPLAY_WIDTH / w_orig, MAX_DISPLAY_HEIGHT / h_orig)
            new_w = int(w_orig * scaling_factor)
            new_h = int(h_orig * scaling_factor)
            resized_frame = cv2.resize(annotated_frame, (new_w, new_h))
            # 7. Hiển thị kết quả
            cv2.imshow("YOLO Gear Detection - BTL Python", resized_frame)
            # 8. Thoát chương trình nếu nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Đang đóng chương trình...")
                break
        else:
            print("Đã chạy hết video.")
            break
    # 9. Giải phóng bộ nhớ và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()
    print("Đã hoàn thành.")
if __name__ == "__main__":
    main()