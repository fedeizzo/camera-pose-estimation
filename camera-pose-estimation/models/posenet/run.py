# https://github.com/ElephantGit/pytorch-posenet/blob/master/train.py

import argparse
import random
import numpy as np
import torch

from config_parser import ConfigParser
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, SGD

from dataset import LowMemoryDataset, HighMemoryDataset
from model import get_posenet
from train import train


def weighted_mse_loss(weight: torch.tensor):
    def fun(input, target):
        return (weight * (input - target) ** 2).mean()

    return fun


def set_random_seed(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_model(config_model, device):
    if config_model["name"] == "posenet":
        model = get_posenet(config_model["outputs"]).to(device)
    else:
        raise ValueError(f"Unknown model: {config_model['name']}")

    return model


def get_dataloader(config_dataloader, config_paths, device, is_train) -> DataLoader:
    dataset = HighMemoryDataset(
        config_paths["dataset"],
        config_paths["images"],
        device,
        is_train=is_train
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config_dataloader["batch_size"],
        shuffle=True
    )

    return train_loader


def get_loss(config_loss, device):
    if config_loss["type"] == "mse":
        criterion = torch.nn.MSELoss()
    elif config_loss["type"] == "weighted":
        criterion = weighted_mse_loss(torch.Tensor(config_loss["weights"]).to(device))

    return criterion


def get_optimizer(config_optimizer, paramters):
    if config_optimizer["name"] == "SGD":
        optimizer = SGD(
            paramters,
            lr=config_optimizer["lr"],
            momentum=config_optimizer["momentum"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config_optimizer['name']}")

    return optimizer


def get_scheduler(config_scheduler, optimizer):
    if config_scheduler["name"] == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=config_scheduler["step_size"],
            gamma=config_scheduler["gamma"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {config_scheduler['name']}")

    return scheduler


def main(config_path: str):
    config = ConfigParser(config_path)
    device = get_device()
    set_random_seed(config["environment"]["seed"])

    train_loader = get_dataloader(config["dataloader"], config["paths"], device, is_train=True)
    criterion = get_loss(config["loss"], device)
    model = get_model(config["model"], device)

    # only parameters of final layer are being optimized
    optimizer = get_optimizer(config["optimizer"], model.fc.parameters())
    scheduler = get_scheduler(config["scheduler"], optimizer)

    train(model, train_loader, criterion, optimizer, scheduler, config["environment"]["epochs"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Base model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file")
    parser.add_argument("-t", "--train", action="store_true", help="Train model flag")
    parser.add_argument(
        "-i", "--inference", action="store_true", help="Inference model flag"
    )
    parser.add_argument("-e", "--test", action="store_true", help="Test model flag")

    args = parser.parse_args()
    config_path = args.config

    if sum([args.train, args.inference, args.test]) != 1:
        parser.error("Either --train, --inference, or --test must be provided")

    if args.train:
        main(config_path)
    elif args.inference:
        raise NotImplementedError("Inference mode not yet implemented")
    elif args.test:
        raise NotImplementedError("Test mode not yet implemented")
