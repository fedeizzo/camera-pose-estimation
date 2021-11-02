import torch.nn as nn

from torchvision import models


def get_posenet(outputs: int) -> nn.Module:
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, outputs),
    )

    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, outputs),
    # )

    return model
