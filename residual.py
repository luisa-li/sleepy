import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

class Residual(nn.Module):
    """The Residual block of ResNet models from the d2l textbook"""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, dp=0.1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

        # added a dropout layer
        self.dp = nn.Dropout(p=dp)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X

        # dropout layer
        # return F.relu(Y)
        return self.dp(F.relu(Y))
        
class ResNet(nn.Module):

    def __init__(self, arch, lr=0.1, num_classes=2, dp=0.1):
        super(ResNet, self).__init__()

        self.dp = dp
        self.lr = lr

        self.net = nn.Sequential(self.b1())

        for i, b in enumerate(arch):
            # residual blocks
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
            # flattening and final linear layer to the needed output size
        self.net.add_module('last', 
            nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.LazyLinear(num_classes)))

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2, dp=self.dp))
            else:
                blk.append(Residual(num_channels, dp=self.dp))
        return nn.Sequential(*blk)
    

    def forward(self, X):
        return self.net(X)
    
    
class ResNet18(ResNet):
    def __init__(self, lr=0.01, num_classes=2, dp=0.1):
        self.dp = dp
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)), lr, num_classes, dp)

class ResNetSimple(ResNet):
    def __init__(self, lr=0.01, num_classes=2, dp=0.1):
        self.dp = dp
        super().__init__(((2, 64), (2, 128)), lr, num_classes, dp)
