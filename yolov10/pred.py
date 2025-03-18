import sys
# sys.path.insert(0, '/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo')
from ultralytics import YOLO

# 載入 YOLOv10 模型
model = YOLO('/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/dataset/best.pt')
img1 ="/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/dataset/train/images/2024_12_17_17_03_55.mp4_1.jpg"
img2 ="/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/dataset/fusion/images/2024_12_17_17_03_55.mp4_0.jpg" 
# 讀取圖片並進行推論
results = model(img1,x2=img2)

# 顯示結果
results[0].show()

# 儲存結果
results[0].save('output.jpg')
