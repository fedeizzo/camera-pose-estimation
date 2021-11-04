# https://github.com/ElephantGit/pytorch-posenet/blob/master/train.py

import argparse
import random
import numpy as np
import torch
import os

from config_parser import ConfigParser
from torch.optim import lr_scheduler, SGD
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable

from dataset import LowMemoryDataset, HighMemoryDataset
from model import MeNet
from train import train
from aim import Run

from os import makedirs, getcwd, chdir, system, chmod
from os.path import isdir, join
from shutil import copy


def create_experiment_dir(net_weights_dir: str, experiment_name: str) -> str:
    experiment_dir = join(net_weights_dir, experiment_name)
    makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_config_ro(config_src: str, config_dst: str):
    copy(config_src, config_dst)
    chmod(config_dst, 0o444)


def weighted_mse_loss(
    weight: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def fun(input: torch.Tensor, target: torch.Tensor):
        return (weight * (input - target) ** 2).mean()

    return fun


def dense_custom_loss(
    alpha: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def fun(input: torch.Tensor, target: torch.Tensor):
        return torch.sqrt(
            torch.sum((target[:, :3] - input[:, :3]) ** 2)
        ) + alpha * torch.sqrt(torch.sum((target[:, 3:] - input[:, 3:]) ** 2))

    return fun


def set_random_seed(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(config_model, device: torch.device) -> torch.nn.Module:
    if config_model["name"] == "menet":
        model = MeNet(config_model["outputs"], 16)
    else:
        raise ValueError(f"Unknown model: {config_model['name']}")

    return model


def get_dataloader(
    dataset_path: str,
    images_path: str,
    device: torch.device,
    is_train: bool,
    dataset_type,
    batch_size: Optional[int],
) -> DataLoader:
    dataset = dataset_type(
        dataset_path,
        images_path,
        device,
        is_train=is_train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size else 1,
        shuffle=True if batch_size else False,
    )


def get_dataloaders(
    train_dataset_path: str,
    validation_dataset_path: str,
    test_dataset_path: str,
    images_path: str,
    batch_size: int,
    dataset_type: str,
    device: torch.device,
) -> Dict[str, DataLoader]:
    dataloaders = {}
    for phase, dataset_path in zip(
        ["train", "validation", "test"],
        [train_dataset_path, validation_dataset_path, test_dataset_path],
    ):
        dataloaders[phase] = get_dataloader(
            dataset_path,
            images_path,
            device,
            phase == "train",
            HighMemoryDataset if dataset_type == "HighMemory" else LowMemoryDataset,
            batch_size if phase != "test" else None,
        )
    return dataloaders


def get_loss(config_loss: dict, device: torch.device):
    if config_loss["type"] == "mse":
        criterion = torch.nn.MSELoss()
    if config_loss["type"] == "dense_custom":
        criterion = dense_custom_loss(alpha=1)
    elif config_loss["type"] == "weighted":
        criterion = weighted_mse_loss(torch.Tensor(config_loss["weights"]).to(device))

    return criterion


def get_optimizer(config_optimizer: dict, paramters) -> torch.optim.Optimizer:
    if config_optimizer["name"] == "SGD":
        optimizer = SGD(
            paramters,
            lr=config_optimizer["lr"],
            momentum=config_optimizer["momentum"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config_optimizer['name']}")

    return optimizer


def get_scheduler(config_scheduler: dict, optimizer: torch.optim.Optimizer):
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
    experiment_dir = create_experiment_dir(
        config["paths"]["net_weights_dir"],
        config["environment"]["experiment_name"],
    )

    aim_run = Run(
        repo=config["paths"]["aim_dir"],
        experiment=config["environment"]["experiment_name"],
        run_hash=config["environment"]["run_name"],
    )
    aim_run[...] = config.get_config()
    save_config_ro(
        config_path,
        os.path.join(experiment_dir, config["environment"]["run_name"] + "_config.ini"),
    )

    train_dataset_path = config["paths"]["train_dataset"]
    validation_dataset_path = config["paths"]["validation_dataset"]
    test_dataset_path = config["paths"]["test_dataset"]
    images_path = config["paths"]["images"]
    batch_size = config["dataloader"]["batch_size"]
    dataset_type = config["dataloader"]["dataset_type"]
    dataloaders = get_dataloaders(
        train_dataset_path,
        validation_dataset_path,
        test_dataset_path,
        images_path,
        batch_size,
        dataset_type,
        device,
    )
    # test_dataloader = dataloaders["test"]
    # del dataloaders["test"]

    criterion = get_loss(config["loss"], device)
    model = get_model(config["model"], device)

    # only parameters of final layer are being optimized
    optimizer = get_optimizer(config["optimizer"], model.fc.parameters())
    scheduler = get_scheduler(config["scheduler"], optimizer)

    train(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        config["environment"]["epochs"],
        aim_run,
    )


if __name__ == "__main__":
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
