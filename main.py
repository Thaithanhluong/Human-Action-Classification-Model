from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import os

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model("./model/model_v1.keras")
    print("------------------------------ Model được tải thành công ------------------------------")
except Exception as e:
    print(f"------------------------------ Đã có lỗi khi tải Model: {e} ------------------------------")    
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results):
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
        keypoints.append(lm.z)
        keypoints.append(lm.visibility)
    return keypoints

@app.route('/')
def home():
    actions = get_Actions("./data")
    return render_template('index.html', actions=actions)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the list of uploaded videos
    upload_dir = os.path.join('static', 'uploads')
    uploaded_videos = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    
    if not uploaded_videos:
        return jsonify({'error': 'No video found'}), 400

    # Sort videos by modification time and get the latest one
    uploaded_videos.sort(key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)
    latest_video = uploaded_videos[0]

    video_path = os.path.join(upload_dir, latest_video)

    cap = cv2.VideoCapture(video_path)
    lm_list = []
    n_time_steps = 10

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            keypoints = extract_keypoints(results)
            lm_list.append(keypoints)

            if len(lm_list) > n_time_steps:
                lm_list.pop(0)

            if len(lm_list) == n_time_steps:
                input_data = np.expand_dims(np.array(lm_list), axis=0)
                predictions_array = model.predict(input_data, verbose=0)
                max_index = np.argmax(predictions_array)
                actions = get_Actions("./data")
                predicted_action = actions[max_index] if max_index < len(actions) else "UNKNOWN"
                confidence = predictions_array[0][max_index] * 100

                # Reset lm_list for the next 10 frames
                lm_list = []

                # Return the result for the current 10 frames
                return jsonify({'action': predicted_action, 'confidence': confidence})

    cap.release()
    return jsonify({'error': 'Failed to process video'}), 500

@app.route('/delete_video', methods=['POST'])
def delete_video():
    data = request.get_json()
    file_name = data.get('fileName')
    if not file_name:
        return jsonify({'error': 'No file name provided'}), 400

    video_path = os.path.join('static', 'uploads', file_name)
    if os.path.exists(video_path):
        os.remove(video_path)
        return jsonify({'message': 'Video deleted successfully'})
    else:
        return jsonify({'error': 'Video not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Save the uploaded video to a temporary file
    video_path = os.path.join('static', 'uploads', file.filename)
    file.save(video_path)

    return jsonify({'message': 'Video uploaded successfully'})

def get_Actions(data_dir="./data"):
    if not os.path.exists(data_dir):
        return []
    unique_actions = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return unique_actions

if __name__ == '__main__':
    app.run(debug=True)