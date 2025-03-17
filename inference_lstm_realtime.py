import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import os

# ✅ 1. Tự động lấy danh sách hành động từ dữ liệu
def get_Actions(data_dir="./data"):
    """Lấy danh sách các hành động từ thư mục dữ liệu"""
    if not os.path.exists(data_dir):
        print(f"⚠️ Thư mục {data_dir} không tồn tại! Sử dụng danh sách mặc định.")
        return []

    unique_actions = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    if not unique_actions:
        print("⚠️ Không tìm thấy hành động nào trong thư mục dữ liệu!")
    
    return unique_actions

# ✅ 2. Lấy danh sách hành động từ thư mục
actions = get_Actions("./data")
if not actions:
    actions = ['UNKNOWN']  # Nếu không có dữ liệu, đặt nhãn mặc định

print("📌 Danh sách hành động:", actions)

# ✅ 3. Khởi tạo Mediapipe & Model
label = "Warmup..."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Tải mô hình
try:
    model = tf.keras.models.load_model("./model/model_v1.keras")
except Exception as e:
    print(f"❌ Không thể tải mô hình: {e}")
    exit()

cap = cv2.VideoCapture(0)

def make_landmark_timestep(results):
    """Trích xuất các đặc trưng từ khung xương"""
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    """Vẽ các điểm mốc lên ảnh"""
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    h, w, _ = img.shape
    for lm in results.pose_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img

def draw_class_on_image(label, img):
    """Vẽ nhãn dự đoán lên ảnh"""
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def detect(model, lm_list):
    """Nhận diện hành động từ mô hình"""
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list, verbose=0)
    max_index = np.argmax(results[0])
    label = actions[max_index] if max_index < len(actions) else "UNKNOWN"
    return label

# ✅ 4. Chạy vòng lặp webcam
i = 0
warmup_frames = 60

while True:
    success, img = cap.read()
    if not success:
        print("❌ Lỗi khi đọc webcam!")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    i += 1
    if i > warmup_frames:
        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                threading.Thread(target=detect, args=(model, lm_list)).start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
