import torch.nn as nn

from torchvision import models

def get_posenet(outputs: int) -> nn.Module:
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, outputs)

    return model