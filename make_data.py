###1. **Import necessary libraries**:
import cv2
import mediapipe as mp
import pandas as pd
import os
import re

###2. **Initialize webcam and mediapipe**:
cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

###3. **Define constants and variables**:
lm_list = []
label = "WAVE_HAND"
no_of_frames = 600

###4. **Define function to extract landmarks**:
def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

###5. **Define function to draw landmarks on image**:
def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

###6. **Define function to generate unique filenames**:
def get_unique_filename(directory, base_filename, extension):
    filename = f"{base_filename}.{extension}"
    counter = 1
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base_filename}_{counter}.{extension}"
        counter += 1
    return filename

###7. **Capture frames and process landmarks**:
while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            frame = draw_landmark_on_image(mpDraw, results, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

###8. **Create necessary directories**:
if not os.path.exists('data'):
    os.makedirs('data')
label_dir = os.path.join('data', label)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

###9. **Save landmarks to CSV file**:
df = pd.DataFrame(lm_list)
unique_filename = get_unique_filename(label_dir, label, "txt")
df.to_csv(os.path.join(label_dir, unique_filename), index=False)

###10. **Release resources**:
cap.release()
cv2.destroyAllWindows()
