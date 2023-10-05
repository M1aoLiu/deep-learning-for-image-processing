import os
import random
from PIL import Image
import shutil

DATA_DIR = "../CamVid"
random.seed(123)
val_rate = 0.3
train_image = "train_images"
train_label = "train_labels"
val_image = "val_images"
val_label = "val_labels"


def rename(root):
    """
    将labels中的图像名称_L去掉
    Args:
        root: 图像根路径

    Returns:

    """
    mask_dir = os.path.join(root, "labels")
    if not os.path.exists(mask_dir):
        raise ValueError("Mask dir does not exist!")
    # 获取mask的所有名称
    mask_list = os.listdir(mask_dir)
    for mask in mask_list:
        # 替换名称
        new_name = mask.replace("_L.png", ".png")
        # 修改名称
        os.rename(os.path.join(mask_dir, mask), os.path.join(mask_dir, new_name))


def split_data(root):
    """
    划分数据集train:val = 7：3
    Args:
        root:

    Returns:

    """
    # 判断目录是否存在
    if not os.path.exists(os.path.join(root, train_image)):
        os.mkdir(os.path.join(root, train_image))
    if not os.path.exists(os.path.join(root, train_label)):
        os.mkdir(os.path.join(root, train_label))
    if not os.path.exists(os.path.join(root, val_image)):
        os.mkdir(os.path.join(root, val_image))
    if not os.path.exists(os.path.join(root, val_label)):
        os.mkdir(os.path.join(root, val_label))

    mask_dir = os.path.join(root, "labels")
    mask_list = os.listdir(mask_dir)
    image_dir = os.path.join(root, "images")
    image_list = os.listdir(image_dir)

    # 如果不存在，报错
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError("Mask dir or image dir does not exist!")

    num = len(mask_list)
    val_index = random.sample(range(0, num), k=int(num * val_rate))
    for i in range(num):
        # 将image和label分别移动到val目录下
        if i in val_index:
            shutil.move(os.path.join(image_dir, image_list[i]),
                        os.path.join(root, "val_images", image_list[i]))  # 移动图像
            shutil.move(os.path.join(mask_dir, mask_list[i]),
                        os.path.join(root, "val_labels", mask_list[i]))  # 移动图像
        else:  # 移动到train目录下
            shutil.move(os.path.join(image_dir, image_list[i]),
                        os.path.join(root, "train_images", image_list[i]))
            shutil.move(os.path.join(mask_dir, mask_list[i]),
                        os.path.join(root, "train_labels", mask_list[i]))

    print("split_data OK!")
