import os
import shutil
from tqdm import tqdm

video_dir = "/data/jlt/CSL/frames_512x512/"
new_video_dir = "/data/jlt/CSL/frames_512x512_split/"
split_file = "/data/jlt/CSL/label/split_1.txt"

split_dict = {"train":[], "test":[], "dev":[]}
with open(split_file, 'r') as file:
    for line in file:
        line = line.strip()  # 去除行末尾的换行符和空格
        video, split = line.split('|')  # 使用特定字符进行分割
        if split == 'split':
            continue

        split_dict[split].append(video)  # 将分割后的结果添加到字典中

split_dataset = list(split_dict.keys())
for set_split in split_dataset:
    now_split_dir = new_video_dir + set_split
    os.makedirs(now_split_dir, exist_ok=True)
    for single_video in tqdm(split_dict[set_split]):
        src_dir = video_dir + single_video
        shutil.move(src_dir, now_split_dir)
