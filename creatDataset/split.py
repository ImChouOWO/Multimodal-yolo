import os
import random
import shutil
from colorama import Fore, Back, Style
from tqdm import tqdm 
from pathlib import Path
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
    folder_list = [
    "train/images", "train/labels",
    "val/images", "val/labels",
    "test/images", "test/labels",
    "fusion/images"
    ]

    print(Fore.LIGHTBLUE_EX + "Creating dataset folders..." + Style.RESET_ALL)

    for sub_folder in tqdm(folder_list, desc="Creating folders", unit=" folder"):
        os.makedirs(os.path.join(folder, sub_folder), exist_ok=True)
    print(Fore.GREEN+"Finish creatting folder"+Style.RESET_ALL)

def split_dataset(folder, fusion,proportion_list:list, dataset ="/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/dataset"):
    
    print(Fore.LIGHTBLUE_EX+ f"Check {dataset}"+Style.RESET_ALL)
    if not os.path.exists(dataset):
        os.makedirs(dataset)
        print(Fore.BLUE+"Folder not exists, creatting folder"+Style.RESET_ALL)
    else:
        print(Fore.GREEN+"Folder exists")

    print(Fore.LIGHTBLUE_EX+f"check {folder}"+Style.RESET_ALL)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(Fore.BLUE+"Folder not exists, creatting folder"+Style.RESET_ALL)
    else:
        print(Fore.GREEN+"Folder exists")
    files = os.listdir(folder)
    random.shuffle(files)
    total = len(files)
    training_set_count = int(total*(proportion_list[0]/sum(proportion_list)))
    val_set_count = int(total*(proportion_list[1]/sum(proportion_list)))
    test_set_count = int(total-training_set_count-val_set_count)
    print(Fore.YELLOW+f"training set count:{training_set_count}, val set count:{val_set_count}, test set count:{test_set_count}"+Style.RESET_ALL)


    # 分配資料
    train_files = files[:training_set_count]
    val_files = files[training_set_count:training_set_count + val_set_count]
    test_files = files[training_set_count + val_set_count:]  # 剩餘的直接當測試集
    # 複製檔案到對應資料夾
    print(Fore.LIGHTBLUE_EX + "Copy files to training set folder..." + Style.RESET_ALL)
    for file in tqdm(train_files, desc="Copy files", unit=" folder"):

        shutil.copy(os.path.join(folder, file), os.path.join(dataset, f"train/images/{file}"))
    print(Fore.LIGHTBLUE_EX + "Copy files to val set folder..." + Style.RESET_ALL)

    for file in tqdm(val_files, desc="Copy files", unit=" folder"):
        shutil.copy(os.path.join(folder, file), os.path.join(dataset, f"val/images/{file}"))

    print(Fore.LIGHTBLUE_EX + "Copy files to test set folder..." + Style.RESET_ALL)
    for file in tqdm(test_files, desc="Copy files", unit=" folder"):
        shutil.copy(os.path.join(folder, file), os.path.join(dataset, f"test/images/{file}"))

    print(Fore.LIGHTBLUE_EX+"Copy files to fusion set folder..."+Style.RESET_ALL)
    fusion_files = os.listdir(fusion)
    for file in tqdm(fusion_files, desc="Copy files", unit=" folder"):
        shutil.copy(os.path.join(fusion, file), os.path.join(dataset, f"fusion/images"))

def copy_lable(folder, label_folder):
    folder_list = [
    "train/labels",
    "val/labels",
     "test/labels"]
    print(Fore.LIGHTBLUE_EX+"copy labes to training set folder"+Style.RESET_ALL)
    for file in tqdm(os.listdir(os.path.join(folder, "train/images")),desc="Copy training label:"):
        file = Path(file).stem
        shutil.copy(os.path.join(label_folder, f"{file}.txt"), os.path.join(folder, f"train/labels/{file}.txt"))

    print(Fore.LIGHTBLUE_EX+"copy labes to val set folder"+Style.RESET_ALL)
    for file in tqdm(os.listdir(os.path.join(folder, "val/images")),desc="Copy val label:"):
        file = Path(file).stem
        shutil.copy(os.path.join(label_folder, f"{file}.txt"), os.path.join(folder, f"val/labels/{file}.txt"))

    print(Fore.LIGHTBLUE_EX+"copy labes to test set folder"+Style.RESET_ALL)
    for file in tqdm(os.listdir(os.path.join(folder, "test/images")),desc="Copy test label:"):
        file = Path(file).stem
        shutil.copy(os.path.join(label_folder, f"{file}.txt"), os.path.join(folder, f"test/labels/{file}.txt"))
    


if __name__ == "__main__":
    dataset_folder = "/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/dataset"
    img_folder = "/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/output/Visual"
    thermal_folder = "/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/output/Thermal"
    label_folder = "/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/creatDataset/dataset/label"
    creat_folder(folder=dataset_folder)
    split_dataset(folder=img_folder, dataset=dataset_folder, proportion_list=[90,5,5], fusion=thermal_folder)
    copy_lable(folder=dataset_folder ,label_folder = label_folder)
    print(Fore.GREEN+"All finished"+Style.RESET_ALL)