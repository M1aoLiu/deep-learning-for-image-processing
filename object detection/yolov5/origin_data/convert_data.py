#  -*- coding: UTF-8 -*
import os
import json
import random
import shutil

"""
将train和val的json文件转换为coco需要的txt文件
"""


def json_to_txt(in_path, out_path):
    """
    将json文件转换成coco需要的txt文件
    Args:
        in_path: 输入路径
        out_path: 输出路径

    Returns:

    """
    dataset_list = ["train", "val"]
    if not os.path.exists(out_path):
        print("create directory:{}".format(out_path))
        os.mkdir(out_path)

    for mode in dataset_list:
        # 创建txt的输出目录
        if not os.path.exists(out_path + mode + "/"):
            print("create directory:{}".format(out_path + mode + "/"))
            os.mkdir(out_path + mode + "/")
        with open(in_path + "TT100K_CoCo_format_45_" + mode + ".json", 'r') as f:
            json_file = json.load(f)
        annotations = json_file["annotations"]
        images = json_file["images"]
        i = 0
        for image in images:
            file_name = image['file_name']
            id = image['id']
            weight = image['width']
            height = image['height']
            for annotation in annotations:
                bbox = annotation['bbox']
                category_id = annotation['category_id']
                image_id = annotation['image_id']
                if image_id == id:
                    # 归一化
                    dw = 1. / weight
                    dh = 1. / height
                    x = (bbox[0] + bbox[2] / 2) * dw
                    y = (bbox[1] + bbox[3] / 2) * dh
                    w = bbox[2] * dw
                    h = bbox[3] * dh
                    f = open(out_path + mode + "/" + str(file_name[:-4]) + ".txt", 'a')
                    f.write("{} {} {} {} {}\n".format(str(category_id - 1), str(round(x, 6)), str(round(y, 6)),
                                                      str(round(w, 6)), str(round(h, 6))))

            i += 1
            if i % 100 == 0:
                print(i)
        print("dataset:{} finished the transfer!".format(mode))


def move_images(in_path, out_path):
    """
    根据生成的txt文件移动图片到相应路径
    Args:
        in_path: txt的路径
        out_path: 图片输出路径

    Returns:

    """
    dataset_list = ["train", "val"]
    if not os.path.exists(out_path):
        print("create directory:{}".format(out_path))
        os.mkdir(out_path)
    for mode in dataset_list:
        # 创建image的输出目录
        if not os.path.exists(out_path + mode + "/"):
            print("create directory:{}".format(out_path + mode + "/"))
            os.mkdir(out_path + mode + "/")
        image_list = os.listdir(in_path + mode)
        image_ids = [i[:-4] + ".jpg" for i in image_list]
        for image_id in image_ids:
            # origin_data/mode.jpg -> ../mydata/mode/.jpg
            shutil.copy("./" + mode + "/" + image_id, out_path + mode)



if __name__ == '__main__':
    json_to_txt("./annotations/", "../mydata/labels/")
    move_images("../mydata/labels/", "../mydata/images/")
