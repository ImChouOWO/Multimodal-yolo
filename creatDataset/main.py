import cv2
import os

video_path ="/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/video/Visual"
files_name = os.listdir(video_path)
output_folder = '/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/output/Visual'
os.makedirs(output_folder, exist_ok=True)
for file in files_name:
    target = os.path.join(video_path, file)
    cap =cv2.VideoCapture(target)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps
    print(f"{file} info \n fps:{fps } \n frames:{frames} \n total sec:{int(duration)}")

    frame_count = 0
    sec = 0

    while cap.isOpened:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % fps ==0:
            frame_files = os.path.join(output_folder, f"{file}_{sec}.jpg")
            cv2.imwrite(frame_files, frame)
            print(f"save {frame_files}")
            sec +=1
        frame_count +=1
    cap.release()
    print("down")