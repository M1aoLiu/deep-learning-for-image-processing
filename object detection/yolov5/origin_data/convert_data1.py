#  -*- coding: UTF-8 -*
import os
import json
import random
import shutil


def get_top_45_categories(ann_path):
    """
    获取前45类标签
    Args:
        ann_path: annotation.json的路径

    Returns:

    """
    categories_dict = {}
    top_45_list = []
    with open(ann_path, 'r') as f:
        json_file = json.load(f)
    imgs = json_file["imgs"]
    for img_id, msgs in imgs.items():
        path = msgs["path"]
        # if path.split("/")[0] == "train" or path.split("/")[0] == "test":
        objects = msgs["objects"]
        for obj in objects:
            category = obj["category"]
            if category not in categories_dict:
                categories_dict[category] = 1
            else:
                categories_dict[category] += 1

    categories_tuple = sorted(categories_dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(45):
        top_45_list.append(categories_tuple[i])
    print(top_45_list)


if __name__ == '__main__':
    get_top_45_categories("./annotations/annotations.json")
