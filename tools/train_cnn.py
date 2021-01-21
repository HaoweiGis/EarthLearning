import os,sys
import argparse
import numpy as np
from datetime import datetime
from numpy import core

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from core.data.dataloador import get_cnn_point_dataset
from core.models.machineLearning.osanet import TwoNet
from core.utils.cnn_point import validate, show_confMat


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert shp to semantic segmentation datasets')
    parser.add_argument('--input', default=r'input.csv' ,help='raster data path' )
    parser.add_argument('--testsize', default=0.3 ,help='raster data path' )
    parser.add_argument('--batchsize', default=20 ,help='raster data path' )
    parser.add_argument('--epoch', default=50 ,help='raster data path' )
    parser.add_argument('--output', default=r'pred_rf.tif', help='output path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    EPOCH = args.epoch
    BATCH_SIZE = args.batchsize
    classes_name = [str(c) for c in range(5)] # 分类地物数量

    dataset = 'point_cnn'
    train_dataset = get_cnn_point_dataset(dataset, split='train')
    val_dataset = get_cnn_point_dataset(dataset, split='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # 检查cuda是否可用
    use_cuda = torch.cuda.is_available()
    # use_cuda = None
    # 生成log
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "log")
    log_dir = os.path.join(log_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)


    # ---------------------搭建网络--------------------------
    twonet = TwoNet(9, 128, 1)  # 创建CNN
    twonet.init_weights()  # 初始化权值
    twonet = twonet.double()

    # --------------------设置损失函数和优化器----------------------
    optimizer = optim.Adam(twonet.parameters())  # lr:(default: 1e-3)优化器
    criterion = nn.CrossEntropyLoss()  # 损失函数
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=EPOCH/2, gamma=0.5)  # 设置学习率下降策略

    # --------------------训练------------------------------
    if(use_cuda):  # 使用GPU
        twonet = twonet.cuda()
    for epoch in range(EPOCH):
        print("epoch:{}".format(epoch+1))
        loss_sigma = 0.0    # 记录一个epoch的loss之和
        correct = 0.0
        total = 0.0
        scheduler.step()  # 更新学习率

        for batch_idx, (image, label) in enumerate(train_loader):
            # 获取图片和标签

            if(use_cuda):
                image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()  # 清空梯度
            twonet = twonet.train()
            outputs = twonet(image)
            loss = criterion(outputs, label)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权值

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += ((predicted == label).squeeze().sum()).item()
            loss_sigma += loss.item()

            # 每 BATCH_SIZE 个 iteration 打印一次训练信息，loss为 BATCH_SIZE 个 iteration 的平均
            if batch_idx % BATCH_SIZE == BATCH_SIZE-1:
                loss_avg = loss_sigma / BATCH_SIZE
                loss_sigma = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, EPOCH, batch_idx + 1, len(train_loader), loss_avg, correct / total))
                # 记录训练loss
                writer.add_scalars(
                    'Loss_group', {'train_loss': loss_avg}, epoch)
                # 记录learning rate
                writer.add_scalar(
                    'learning rate', scheduler.get_lr()[0], epoch)
                # 记录Accuracy
                writer.add_scalars('Accuracy_group', {
                                'train_acc': correct / total}, epoch)
        # 每个epoch，记录梯度，权值
        # for name, layer in twonet.named_parameters():
        #     writer.add_histogram(
        #         name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        #     writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        if epoch % 1 == 0:
            loss_sigma = 0.0
            cls_num = len(classes_name)
            conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
            twonet.eval()
            for batch_idx, (data1, data2) in enumerate(test_loader):
                # for batch_idx, data in enumerate(test_loader):
                input1, labels = data1,data2   # inputs.shape → torch.Size([18, 46, 5, 5])
                # input2, label2 = data2   # inputs.shape
                if(use_cuda):
                    input1, labels = input1.cuda(), labels.cuda()
                    # input2, label2 = input2.cuda(), label2.cuda()
                twonet = twonet.train()
                outputs = twonet(input1)
                outputs.detach_()  # 不求梯度
                loss = criterion(outputs, labels)
                loss_sigma += loss.item()

                _, predicted = torch.max(outputs.data, 1)  # 统计
                # labels = labels.data    # Variable --> tensor
                # 统计混淆矩阵
                for j in range(len(labels)):
                    cate_i = labels[j]
                    pre_i = predicted[j]
                    conf_mat[cate_i, pre_i] += 1.0
            print('{} set Accuracy:{:.2%}'.format(
                'Valid', conf_mat.trace() / conf_mat.sum()))
            # 记录Loss, accuracy
            writer.add_scalars(
                'Loss_group', {'valid_loss': loss_sigma / len(test_loader)}, epoch)
            writer.add_scalars('Accuracy_group', {
                            'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
    print('Finished Training')

    # ----------------------- 保存模型 并且绘制混淆矩阵图 -------------------------
    # twonet_save_path = os.path.join(log_dir, 'net_params.pkl')
    # torch.save(twonet.state_dict(), twonet_save_path)

    conf_mat_train, train_acc = validate(twonet, train_loader, 'train', classes_name)
    conf_mat_valid, valid_acc = validate(twonet, test_loader, 'test', classes_name)

    show_confMat(conf_mat_train, classes_name, 'train', log_dir)
    show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
