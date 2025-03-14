# import contextlib
# import glob
# import inspect
# import math
# import os
# import platform
# import re
# import shutil
# import subprocess
# import time
# from importlib import metadata
# from pathlib import Path
# from typing import Optional
# import yaml

# def yaml_load(file="data.yaml", append_filename=False):
#     """
#     Load YAML data from a file.

#     Args:
#         file (str, optional): File name. Default is 'data.yaml'.
#         append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

#     Returns:
#         (dict): YAML data and file name.
#     """
#     assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
#     with open(file, errors="ignore", encoding="utf-8") as f:
#         s = f.read()  # string

#         # Remove special characters
#         if not s.isprintable():
#             s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

#         # Add YAML filename to dict and return
#         data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
#         if append_filename:
#             data["yaml_file"] = str(file)
#         return data
    
# def test():
#     print(yaml_load("/Users/zhouchenghan/python/GPS_IMU/multimodal-yolo/yolov10/ultralytics/cfg/datasets/coco.yaml")["fusion"])

# if __name__ =="__main__":
#     test()


fusion_pbar = {"fusions":[1,2,3]}
fusion_pbar["test"] = [1,2,4]
print(fusion_pbar)