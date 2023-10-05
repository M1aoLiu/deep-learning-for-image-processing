import os
import random


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    # 指定数据路径
    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5

    # 根据"."分割annotation名称得到图像名称并排序
    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name) # 获取总数据个数
    # 使用随机采样，范围是0-数据个数，采样的个数为验证集个数
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    # 枚举，分别将数据划分成训练集和验证集
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "w")
        eval_f = open("val.txt", "w")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
