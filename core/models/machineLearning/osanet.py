import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.rnn import LSTM
# --------------------搭建网络--------------------------------

def optical2cuda(X):
    X = X.cpu()
    newX = np.zeros(
        (X.shape[0], X.shape[1]-2, X.shape[2], X.shape[3]))
    newX[:,:,:,:] = X[:,:-2,:,:]
    newX = torch.from_numpy(newX).double()
    if torch.cuda.is_available():
        newX = newX.cuda()
    return newX

def sar2cuda(X):
    X = X.cpu()
    newX = np.zeros(
        (X.shape[0], 2, X.shape[2], X.shape[3]))
    newX[:,:,:,:] = X[:,-2:,:,:]
    newX = torch.from_numpy(newX).double()
    if torch.cuda.is_available():
        newX = newX.cuda()
    return newX

class SeparableConv2d(nn.Module):  # Depth wise separable conv
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        # 每个input channel被自己的filters卷积操作
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1, bias=False),
        #     nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 1, bias=False),
        #     nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x)
        # .view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y

class SeparableConv2dAttention(nn.Module):  # Depth wise separable conv
    def __init__(self, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        # 每个input channel被自己的filters卷积操作
        super(SeparableConv2dAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.pointwise = nn.Conv2d(
        #     in_channels, in_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x_center =  x[:,:,int((x.shape[2]+1)/2),int((x.shape[3]+1)/2)].view(x.shape[0],x.shape[1],1,1)
        # x_center = self.avg_pool(x)
        b, c , _, _ = x.size()
        x = self.conv1(x)

        # Excitation
        y = self.fc(x_center).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        # x = self.pointwise(x)
        return y

# TwoNet(9, 128, 1)
class TwoNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer):    # lstm的3个参数
        super().__init__()
        # 20, 14, 5, 5
        self.conv1 = nn.Sequential(
            SeparableConv2d(12, 64, 3, 1, 1),  # 1*1卷积核
            nn.ReLU(inplace=True),
            nn.GroupNorm(64, 64)
            # GroupNorm将channel分组，然后再做归一化；
        )

        self.conv2 = nn.Sequential(
            SeparableConv2d(2, 32, 3, 1, 1),  # 1*1卷积核
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, 32)
            # GroupNorm将channel分组，然后再做归一化；
        )

        self.conv3 = nn.Sequential(
            SeparableConv2dAttention(64, 3, 1, 0),  # 1*1卷积核
            nn.ReLU(inplace=True),
            nn.GroupNorm(64, 64)
            # GroupNorm将channel分组，然后再做归一化；
        )

        self.conv4 = nn.Sequential(
            SeparableConv2dAttention(32, 3, 1, 0),  # 1*1卷积核
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, 32)
            # GroupNorm将channel分组，然后再做归一化；
        )

        self.conv5 = SEBlock(96)

        # 分类
        self.classifier = nn.Sequential(
            nn.Linear(864, 1024),
            # nn.Linear(512*5*5, 2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(512, 6),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):

        optical = optical2cuda(input)
        sar = sar2cuda(input)
        
        optical_x1 = self.conv1(optical)
        sar_x1 = self.conv2(sar)        # print(x2.shape) 打印结果为 torch.Size([BATCH_SIZE, 512, 5, 5])

        optical_x2 = self.conv3(optical_x1)
        sar_x2 = self.conv4(sar_x1)

        optical_sar = torch.cat((optical_x2, sar_x2), 1)

        x3 = self.conv5(optical_sar)
        x4 = x3.view(-1, self.numFeatures(x3))  # 特征映射一维展开

        out = self.classifier(x4)
        return out

    def numFeatures(self, x):
        size = x.size()[1:]  # 获取卷积图像的h,w,depth
        num = 1
        for s in size:
            num *= s
            # print(s)
        return num

    def init_weights(self):  # 初始化权值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()