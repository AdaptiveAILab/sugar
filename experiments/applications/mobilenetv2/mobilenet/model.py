from torchvision import models
import torch.nn as nn

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    in_feats = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_feats, num_classes)
    return net
