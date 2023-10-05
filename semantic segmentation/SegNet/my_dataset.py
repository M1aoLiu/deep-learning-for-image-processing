import torch
import torch.nn as nn
import os
import warnings

from torchvision import transforms

warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np


class CamVidDataset(nn.Module):
    """
    自定义Dataset类用于制作数据集
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        """
        初始化函数
        Args:
            images_dir: 图像路径
            masks_dir: mask路径
            transform: 数据转换
        """
        super(CamVidDataset, self).__init__()
        # 判断目录是否存在
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise ValueError("Mask dir or image dir does not exist!")

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        # [32, 3]: 32个类别，3表示RGB
        self.classes = np.load('./CamVid/classes.npy')

    def one_hot(self, image):
        """
        将[H, W, 3]的原始mask图像转换成[H, W, K]的one hot编码形式，K为类别个数
        Args:
            image:

        Returns:

        """
        output_shape = (image.shape[0], image.shape[1], self.classes.shape[0])
        output = np.zeros(output_shape)

        for c in range(self.classes.shape[0]):
            # nanmin忽略nan数据求最小值 在channel=3的维度上求最小值并比较返回布尔数组
            # 这样就能够知道每个类别在哪
            label = np.nanmin(self.classes[c] == image, axis=2)
            # 赋值给output，最终输出[H, W, K]的bool列表
            output[:, :, c] = label

        return output

    def __len__(self):
        """
        返回长度
        Returns:

        """
        return len(self.ids)

    def __getitem__(self, idx):
        """
        根据索引返回image和mask
        Args:
            idx: 索引

        Returns:

        """
        # 读取数据
        # 返回的是[3, H, W]
        image = self.transform(np.array(Image.open(self.images_fps[idx]).convert('RGB')))
        mask = np.array(Image.open(self.masks_fps[idx]).convert('RGB'))
        # 返回的是[K, H, W]
		mask = self.transform(self.one_hot(mask))

        return image, mask


def load_data(DATA_DIR, mode="train"):
    # 数据转换
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Resize([448, 448])
        ]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    x_train_dir = os.path.join(DATA_DIR, 'train_images')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')
    x_valid_dir = os.path.join(DATA_DIR, 'val_images')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

    train_dataset = CamVidDataset(
        x_train_dir,
        y_train_dir,
        data_transform["train"]
    )
    val_dataset = CamVidDataset(
        x_valid_dir,
        y_valid_dir,
        data_transform["val"]
    )
    if mode == "train":
        dataset = train_dataset
    else:
        dataset = val_dataset

    return dataset

# for i, (img, label) in enumerate(train_loader):
#     img = torch.as_tensor(img[0])
#     label = torch.as_tensor(label[0])
#     plt.subplot(121)
#     plt.imshow(img.moveaxis(0, 2))  # 将维度0移动到维度2 即：[C, H, W] 改为 [H, W, C]
#     plt.subplot(122)
#     plt.imshow(label.moveaxis(0, 2))
#
#     plt.show()
