# import sys
# # sys.path.insert(0, '/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo')
# from ultralytics import YOLO

# # 載入 YOLOv10 模型
# model = YOLO('C:/project/Multimodal-yolo/mutimodal/oPv/weights/best.pt')
# img2 ="C:/project/Multimodal-yolo/dataset/VandT/thermal/val/images/010056.jpg"
# img1 ="C:/project/Multimodal-yolo/dataset/VandT/visible/val/images/010056.jpg" 
# # 讀取圖片並進行推論
# results = model(img2,x2=img2)

# # 顯示結果
# results[0].show()

# # 儲存結果
# results[0].save('output.jpg')
import cv2
from ultralytics import YOLO

# 載入模型與圖片
model = YOLO("C:/project/origin_yolov10/run/test/train5/weights/best.pt")
img2_path = "C:/project/Multimodal-yolo/dataset/VandT/thermal/val/images/010056.jpg"
img1_path = "C:/project/Multimodal-yolo/dataset/VandT/visible/val/images/010056.jpg"

# 進行推論（用 thermal 圖片推）
results = model(img1_path)

# 載入你要畫的圖像（可選擇 img1 或 img2）
img = cv2.imread(img1_path)  # 這裡你可以改成 img1_path 看你要畫在哪個圖

# 遍歷每個邊界框
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 邊界框座標
    cls_id = int(box.cls[0])               # 類別編號
    conf = box.conf[0]                     # 置信度

    label = f"{model.names[cls_id]} {conf:.2f}"  # 顯示類別名稱與信心值

    # 畫框與標籤
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

# 顯示並儲存結果
cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('output.jpg', img)
