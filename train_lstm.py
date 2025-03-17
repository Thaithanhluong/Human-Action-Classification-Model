import os
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# ✅ 1. Lấy danh sách hành động từ thư mục dữ liệu
def get_Actions(data_dir="./data"):
    """Lấy danh sách tên thư mục con trong thư mục dữ liệu."""
    unique_actions = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return unique_actions

# Tạo danh sách hành động & bảng ánh xạ
unique_actions = get_Actions("./data")
actions_map = {action: i for i, action in enumerate(unique_actions)}

print("Danh sách hành động:", unique_actions)
print("Bảng ánh xạ:", actions_map)

# ✅ 2. Đọc dữ liệu từ tất cả file trong các thư mục con
def process_data(file_path, label, no_of_timesteps=10):
    """Xử lý dữ liệu từ file và tạo mẫu dữ liệu theo timestep."""
    df = pd.read_csv(file_path)
    dataset = df.values  
    n_samples = len(dataset)

    X, y = [], []
    for i in range(no_of_timesteps, n_samples):
        X.append(dataset[i-no_of_timesteps:i, :])  # Lấy 10 bước thời gian
        y.append(label)  # Gán nhãn

    return X, y

# ✅ 3. Chuẩn bị dữ liệu X, y
X, y = [], []
no_of_timesteps = 10  # Số bước thời gian

for action in unique_actions:
    action_dir = os.path.join("./data", action)
    
    for file in os.listdir(action_dir):
        file_path = os.path.join(action_dir, file)
        X_action, y_action = process_data(file_path, actions_map[action], no_of_timesteps)
        X.extend(X_action)
        y.extend(y_action)

# Chuyển về numpy array
X, y = np.array(X), np.array(y)
print(f"Dữ liệu đầu vào: {X.shape}, Nhãn: {y.shape}")

# ✅ 4. Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ 5. Xây dựng mô hình LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(len(unique_actions), activation="softmax")  # Số lớp output = số hành động
])

# ✅ 6. Compile mô hình
model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")

# ✅ 7. Huấn luyện mô hình
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))

# ✅ 8. Lưu mô hình
model.save("./model/model_v1.keras")
