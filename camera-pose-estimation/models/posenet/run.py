# https://github.com/ElephantGit/pytorch-posenet/blob/master/train.py

import argparse
import random
import numpy as np
import torch
import pickle
import os

from config_parser import ConfigParser
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Callable, Tuple, List

from datasets.absolute import AbsolutePoseDataset, MapNetDataset, SevenScenes
from datasets.relative import RelativePoseDataset
from models.posenet import get_posenet
from models.menet import MeNet
from models.mapnet import MapNet
from criterions.criterions import get_loss
from train import train_model
from test_model import (
    test_model,
    reverse_normalization,
    from_relative_to_absolute_pose,
    compute_absolute_positions,
)
from aim import Run
from torchinfo import summary

from os import makedirs, chmod
from os.path import join
from shutil import copy


def create_experiment_dir(net_weights_dir: str, experiment_name: str) -> str:
    experiment_dir = join(net_weights_dir, experiment_name)
    makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_config_ro(config_src: str, config_dst: str):
    copy(config_src, config_dst)
    chmod(config_dst, 0o444)


def set_random_seed(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(config_model, device: torch.device) -> Tuple[torch.nn.Module, Dataset]:
    if config_model["name"] == "posenet":
        model = get_posenet(config_model["outputs"]).to(device)
        dataset_type = AbsolutePoseDataset
    elif config_model["name"] == "menet":
        model = MeNet(config_model["outputs"]).to(device)
        dataset_type = RelativePoseDataset
    elif config_model["name"] == "mapnet":
        model = MapNet(
            config_model["feature_dimension"], config_model["dropout_rate"]
        ).to(device)
        dataset_type = MapNetDataset
    else:
        raise ValueError(f"Unknown model: {config_model['name']}")

    return model, dataset_type


def get_dataloader(
    dataset_path: str,
    config_dataloader: dict,
    config_paths: dict,
    dataset_type,
    phase: str,
    device: torch.device,
) -> DataLoader:
    if dataset_type == MapNetDataset and phase != "test":
        dataset = MapNetDataset(
            path=dataset_path,
            steps=config_dataloader["step"],
            skip=config_dataloader["skip"],
            color_jitter=config_dataloader["color_jitter"],
            seq=config_dataloader["sequences"][phase],
        )
    elif dataset_type == MapNetDataset and phase == "test":
        dataset = SevenScenes(
            dataset_path=dataset_path,
            seq=config_dataloader["sequences"][phase],
        )
    else:
        dataset = dataset_type(
            dataset_path, config_paths["images"], device, is_train=phase == "train"
        )

    return DataLoader(
        dataset,
        batch_size=config_dataloader["batch_size"],
        shuffle=True,
        num_workers=config_dataloader["num_workers"],
    )


def get_dataloaders(
    config_dataloader: dict,
    config_paths: dict,
    phases: List[str],
    dataset_paths: List[str],
    dataset_type: Dataset,
    device: torch.device,
) -> Dict[str, DataLoader]:
    dataloaders = {}

    assert len(dataset_paths) == len(
        phases
    ), "Number of phases and dataset paths must be equal"

    for phase, dataset_path in zip(phases, dataset_paths):
        dataloaders[phase] = get_dataloader(
            dataset_path, config_dataloader, config_paths, dataset_type, phase, device,
        )
    return dataloaders


def get_optimizer(config_optimizer: dict, paramters) -> torch.optim.Optimizer:
    if config_optimizer["name"] == "SGD":
        optimizer = SGD(
            paramters, lr=config_optimizer["lr"], momentum=config_optimizer["momentum"],
        )
    elif config_optimizer["name"] == "adam":
        optimizer = Adam(
            paramters,
            lr=config_optimizer["lr"],
            weight_decay=config_optimizer["decay"],
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


def train(config_path: str):
    config = ConfigParser(config_path)
    device = get_device()
    set_random_seed(config["environment"]["seed"])
    experiment_dir = create_experiment_dir(
        config["paths"]["net_weights_dir"], config["environment"]["experiment_name"],
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
    # test_dataset_path = config["paths"]["test_dataset"]
    batch_size = config["dataloader"]["batch_size"]

    model, dataset_type = get_model(config["model"], device)
    dataloaders = get_dataloaders(
        config["dataloader"],
        config["paths"],
        ["train", "validation"],
        [train_dataset_path, validation_dataset_path],
        dataset_type,
        device,
    )

    criterion = get_loss(config["loss"], device)

    # only parameters of final layer are being optimized
    if config["model"]["name"] == "posenet":
        optimizer = get_optimizer(config["optimizer"], model.fc.parameters())
    elif config["model"]["name"] == "mapnet":
        param_list = list(model.parameters())
        if criterion.learn_beta:
            param_list.extend([criterion.sax, criterion.saq])
        if criterion.learn_gamma:
            param_list.extend([criterion.srx, criterion.srq])

        optimizer = get_optimizer(config["optimizer"], param_list)
    else:
        optimizer = get_optimizer(config["optimizer"], model.parameters())
    # summary(
    #     model, (batch_size, *dataloaders["train"].dataset[0][0].size()),
    # )
    scheduler = get_scheduler(config["scheduler"], optimizer)

    trained_model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        config["environment"]["epochs"],
        aim_run,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    net_weights_path = os.path.join(
        experiment_dir, config["environment"]["run_name"] + ".pth",
    )
    torch.save(trained_model.state_dict(), net_weights_path)


def test(config_path: str):
    config = ConfigParser(config_path)
    device = get_device()
    experiment_dir = create_experiment_dir(
        config["paths"]["net_weights_dir"], config["environment"]["experiment_name"],
    )
    train_configs = ConfigParser(
        os.path.join(experiment_dir, config["environment"]["run_name"] + "_config.ini")
    )
    set_random_seed(train_configs["environment"]["seed"])

    dataset_path = config["paths"]["dataset"]

    model, dataset_type = get_model(train_configs["model"], device)
    weights_path = os.path.join(
        experiment_dir, config["environment"]["run_name"] + ".pth",
    )
    model.load_state_dict(torch.load(weights_path))
    model = model.to(get_device())

    # TODO
    train_configs["dataloader"]["sequences"] = config["dataloader"]["sequences"]

    dataloaders = get_dataloaders(
        train_configs["dataloader"],
        config["paths"],
        ["test"],
        [dataset_path],
        dataset_type,
        device,
    )

    targets, predictions = test_model(model, dataloaders["test"], device)
    targets.to_csv(config["paths"]["targets"], index=False)
    predictions.to_csv(config["paths"]["predictions"], index=False)

    # quaternion_scaler_path = config["paths"]["quaternion_scaler"]
    # translation_scaler_path = config["paths"]["translation_scaler"]

    # with open(quaternion_scaler_path, "rb") as f:
    #     quaternion_scaler = pickle.load(f)
    # with open(translation_scaler_path, "rb") as f:
    #     translation_scaler = pickle.load(f)

    # predictions = reverse_normalization(
    #     predictions, quaternion_scaler, translation_scaler
    # )
    # predictions.loc[-1] = targets.iloc[0][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]]
    # predictions.index = predictions.index + 1
    # predictions.sort_index(inplace=True)
    # targets = reverse_normalization(targets, quaternion_scaler, translation_scaler)
    # targets = targets[["x", "y", "z"]]
    # predictions = from_relative_to_absolute_pose(predictions)
    # positions = compute_absolute_positions(predictions)
    # import pdb

    # pdb.set_trace()


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
        train(config_path)
    elif args.inference:
        raise NotImplementedError("Inference mode not yet implemented")
    elif args.test:
        test(config_path)
