## Giới thiệu

Đây là đề tài `Nhận diện hành động con người trong video sử dụng mạng học sâu`, là sản phẩm kết thúc môn học `Đồ án ngành`. Đề tài này nhằm mục đích sử dụng mạng học sâu để nhận diện hành động con người trong video.

## Cấu trúc

Dưới đây là mô tả về các tệp tin và thư mục:
### make_data.py 

Tạo dữ liệu huấn luyện bằng cách sử dụng camera thu thâph 600 frame / 1 hành động. 

### train_lstm.py 

Đẩy dữ liệu vào và huấn luyện mô hình.

### inference_lstm_realtime.py

Kiểm thử khả năng suy luận thời gian thực, mô hình sẽ suy luận liên tục với dữ liệu đầu vào từ camera.

### inference_lstm_video.py

Kiểm thử khả năng suy luận của mô hình với đầu vào là video.

### main.py

Tệp tin này chứa mã nguồn chính của đề tài. Nó thực hiện khởi chạy mô hình dự đoán và trang web dự đoán để tương tác với mô hình.

### data/

Thư mục này chứa dữ liệu huấn luyện của đề tài. Bên trong gồm các thư mục con, mỗi thư mục con đại diện cho một nhãn hành động mà mô hình cần dự đoán. Mỗi thư mục con chứa các tệp tin `.txt`, mỗi tệp tin chứa các keypoints được thu thập từ camera bằng Mediapipe. Mỗi tệp tin `.txt` chứa keypoints của 600 frame, mỗi dòng đại diện cho một frame. Cấu trúc mỗi dòng gồm tọa độ của 33 keypoints, mỗi keypoint bao gồm các giá trị x, y, z và visibility.

Ví dụ cấu trúc thư mục:
```
data/
├── clap/
│   ├── clap1.txt
│   ├── clap2.txt
│   └── ...
├── run/

### README.md

Tệp tin này cung cấp thông tin về đề tài, bao gồm hướng dẫn cài đặt và sử dụng.

## Hướng dẫn cài đặt

1. Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

2. Chạy chương trình:
    ```bash
    - Bước 1: chạy python main.py
    - Bước 2: truy cập  http://127.0.0.1:5000/ để vào trang web tương tác với mô hình.
    ```
3. Chức năng chính của trang web:
    - Cho phép nhập vào video cần dự đoán hành động, trả về thông báo là loại hành động nào.
    - Cho phép quay 1 video, sau đó sử dụng video đã quay để dự đoán.
    - Các hành động mô hình có thể dự đoán là: clap (vỗ tay), run (chạy), sit (ngồi), stand (đứng), walk (đi bộ), wave hand (vỗ tay).
## Liên hệ
Họ và tên sinh viên: Thái Thành Lương
Email: 210501014@studen.bdu.edu.vn
Mã số sinh viên: 210501014
Github: https://github.com/thaithanhluong