import torch
import time
import argparse
import numpy as np
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from decoder import SegNet
from my_dataset import load_data


def main(args):
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 模型
    model = SegNet(num_classes=args.num_classes).to(device)
    # print(model)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loss = []
    learning_rate = []  # 保存学习率
    val_map = []  # 保存val的map
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_dataset = load_data(args.data_path, "train")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        total_loss = 0
        for i, (image, mask) in enumerate(train_loader):
            # [B, 3, H, W]
            image = image.to(device)
            # [B, K, H, W] one-hot
            mask = mask.to(device)
            # print(image.shape, mask.shape)
            # [B, num_classes, H, W]
            output = model(image)
            loss = loss_fn(output, mask)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("epoch:{}/{}, train step:{}/{}, train_loss:{}"
                      .format(epoch, args.epochs, i, len(train_dataset), loss.item()))
        train_loss.append(total_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='./CamVid', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=32, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.004, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[5], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练的batch size
    parser.add_argument('--batch-size', default=1, type=int, help='batch size when training.')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
