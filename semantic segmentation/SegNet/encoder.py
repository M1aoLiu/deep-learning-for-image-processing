import torch
import torch.nn as nn
import os


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.stage_1 = self.make_layers(3, 64, 2)
        self.stage_2 = self.make_layers(64, 128, 2)
        self.stage_3 = self.make_layers(128, 256, 3)
        self.stage_4 = self.make_layers(256, 512, 3)
        self.stage_5 = self.make_layers(512, 512, 3)

    def make_layers(self, in_channel, out_channel, conv_num):
        """

        Args:
            in_channel:
            out_channel:
            conv_num:

        Returns:

        """
        layers = []
        layer_1 = [
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        ]
        layer_x = [
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        ]
        layers.extend(layer_1)
        for _ in range(1, conv_num):
            layers.extend(layer_x)

        return nn.Sequential(*layers)

    def forward(self, x):
        # 用来保存各层的池化索引
        pool_indices = []
        x = x.float()

        x = self.stage_1(x)
        # 相当于pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True), pool(data)
        # pool是一个实例化对象（MaxPool2d继承自Module，实现了__call__方法），里面实现了__call__方法，因此可以当成方法使用
        # 分别返回处理好后的数据以及对应的索引位置
        x, pool_indice_1 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_1)

        x = self.stage_2(x)
        x, pool_indice_2 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_2)

        x = self.stage_3(x)
        x, pool_indice_3 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_3)

        x = self.stage_4(x)
        x, pool_indice_4 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_4)

        x = self.stage_5(x)
        x, pool_indice_5 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_5)

        return x, pool_indices