import torch
import torch.nn as nn
from encoder import Encoder
import numpy as np


class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        self.encoder = Encoder()
        # 上采样 从下往上, 1->2->3->4->5
        self.upsample_1 = self.make_layers(512, 512, 3)
        self.upsample_2 = self.make_layers(512, 256, 3)
        self.upsample_3 = self.make_layers(256, 128, 3)
        self.upsample_4 = self.make_layers(128, 64, 2)
        self.upsample_5 = self.make_layers(64, num_classes, 2)

    def make_layers(self, in_channel, out_channel, conv_num):
        """

        Args:
            in_channel:
            out_channel:
            conv_num:

        Returns:

        """
        layers = []
        layer_x = [
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        ]
        last_layer = [
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        ]
        for _ in range(conv_num - 1):
            layers.extend(layer_x)

        layers.extend(last_layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        x, pool_indices = self.encoder(x)

        # 池化索引上采样
        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[4])
        x = self.upsample_1(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[3])
        x = self.upsample_2(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[2])
        x = self.upsample_3(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[1])
        x = self.upsample_4(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[0])
        x = self.upsample_5(x)

        return x

if __name__ == '__main__':
    a = torch.randn(2,2)
    b = a.numpy()
    print(type(b))