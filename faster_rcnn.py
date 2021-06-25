import torch
import torch.nn as nn
# from torch import nn
from torch.nn import functional as F

from torchvision import models


def faster_rcnn():
    # remember about 7x7 first conv, does resnext have it?
    resnet = models.resnet18(pretrained=True)

    backbone = nn.Sequential(*list(resnet.children())[:-2])
    net = FasterRCNN(backbone)
    return net


class RoIPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class RPN(nn.Module):

    def __init__(self):
        super().__init__()

        # TODO: all consts to init

        in_channels = 512
        intermediate_size = in_channels

        self.conv3 = nn.Conv2d(in_channels, intermediate_size, kernel_size=3, stride=1, padding=1)

        N = 9 # TODO: anchors per pixel
        self.reg = nn.Conv2d(intermediate_size, N*4, kernel_size=1, stride=1, padding=0)
        self.cls = nn.Conv2d(intermediate_size, N, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x,):

        x = F.relu(self.conv3(x))

        boxes = self.reg(x)
        classes = self.cls(x)

        return boxes, classes


class FasterRCNN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.rpn = RPN()

    def forward(self, x):
        x = self.backbone(x)

        box, cls = self.rpn(x)

        box = box.permute(0, 2, 3, 1).contiguous()
        box = box.view(box.shape[0], -1, 4)

        # will not work without contiguous ?
        # reshape will
        # view no
        # use reshape?!
        # box = box.reshape(box.shape[0], -1, 4)
        # box = box.reshape(box.shape[0], 4, -1)

        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(cls.shape[0], -1, 1)
         
        return box, cls


# net = faster_rcnn()
# print(net)