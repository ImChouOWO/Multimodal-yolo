import os
import random
import shutil
"""creat dataset folder

dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── fusion/
    └── images/
"""
def creat_folder(folder):
    taget_folder =folder
    os.makedirs(os.path.join(taget_folder, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(taget_folder, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(taget_folder, "val/images"), exist_ok=True)     
    os.makedirs(os.path.join(taget_folder, "val/labels"), exist_ok=True)
    os.makedirs(os.path.join(taget_folder, "test/images"), exist_ok=True)     
    os.makedirs(os.path.join(taget_folder, "test/labels"), exist_ok=True)
    os.makedirs(os.path.join(taget_folder, "fusion/images"), exist_ok=True) 

def split_dataset(folder, proportion_list:list, dataset ="/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/dataset"):
    files = os.listdir(folder)
    random.shuffle(files)
    total = len(files)
    training_set_count = int(total*(proportion_list[0]/sum(proportion_list)))
    val_set_count = int(total*(proportion_list[1]/sum(proportion_list)))
    test_set_count = int(total-training_set_count-val_set_count)
    


    # 分配資料
    train_files = files[:training_set_count]
    val_files = files[training_set_count:training_set_count + val_set_count]
    test_files = files[training_set_count + val_set_count:]  # 剩餘的直接當測試集
    print(train_files)
    # 複製檔案到對應資料夾
    for file in train_files:
        shutil.copy(os.path.join(folder, file), os.path.join(dataset, f"train/images/{file}.jpg"))

    for file in val_files:
        shutil.copy(os.path.join(folder, file), os.path.join(dataset, f"val/images/{file}.jpg"))

    for file in test_files:
        shutil.copy(os.path.join(folder, file), os.path.join(dataset, f"test/images/{file}.jpg"))

    print("檔案分割完成！")

if __name__ == "__main__":
    creat_folder(folder="/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/dataset")
    split_dataset(folder="/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/output/Visual", proportion_list=[90,5,5])