import torch.nn as nn

from torchvision import models


def get_posenet(outputs: int) -> nn.Module:
    model = models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, outputs),
    )

    return model
