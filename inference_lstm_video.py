import cv2,os
import mediapipe as mp
import numpy as np
import tensorflow as tf


# Load model
model = tf.keras.models.load_model("./model/model_v1.keras")

# Define actions list
def get_Actions(data_dir="./data"):
    """Lấy danh sách các hành động từ thư mục dữ liệu"""
    if not os.path.exists(data_dir):
        print(f"⚠️ Thư mục {data_dir} không tồn tại! Sử dụng danh sách mặc định.")
        return []
    unique_actions = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not unique_actions:
        print("⚠️ Không tìm thấy hành động nào trong thư mục dữ liệu!")
    return unique_actions
actions = get_Actions("./data")
if not actions:
    actions = ['UNKNOWN']  # Nếu không có dữ liệu, đặt nhãn mặc định

# print("📌 Danh sách hành động:", actions)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Open video file
video_path = r"D:\Downloads\video.mp4"
cap = cv2.VideoCapture(video_path)

# Parameters
n_time_steps = 10
lm_list = []
last_prediction = None

# Function to extract keypoints
def extract_keypoints(results):
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
        keypoints.append(lm.z)
        keypoints.append(lm.visibility)
    return keypoints

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        keypoints = extract_keypoints(results)
        lm_list.append(keypoints)

        # Nếu số lượng frame vượt quá n_time_steps thì xóa frame cũ nhất
        if len(lm_list) > n_time_steps:
            lm_list.pop(0)

        # Chỉ dự đoán khi đã đủ n_time_steps frame
        if len(lm_list) == n_time_steps:
            input_data = np.expand_dims(np.array(lm_list), axis=0)
            predictions_array = model.predict(input_data, verbose=0)
            max_index = np.argmax(predictions_array)
            confidence = predictions_array[0][max_index] * 100
            current_prediction = f"{actions[max_index]}"

            # Chỉ in nếu dự đoán khác lần trước
            if current_prediction != last_prediction:
                print(f"Predicted action: {current_prediction}")
                last_prediction = current_prediction

cap.release()
