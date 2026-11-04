import torchvision.models as tvm
from torch import nn


def resnet18_cifar(num_classes=10):
    m = tvm.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, num_classes)
    return m
