import argparse
import random
import numpy as np
import torch
import os

from config_parser import ConfigParser
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Any
from PIL import Image

from datasets.absolute import (
    AbsolutePoseDataset,
    MapNetDataset,
    SevenScenes,
    get_image_transform,
)
from datasets.relative import RelativePoseDataset
from models.posenet import get_posenet
from models.menet import MeNet
from models.mapnet import MapNet
from criterions.criterions import get_loss
from train import train_model
from test_model import (
    test_model,
)
from utils.metrics import calculate_MAE_poses
from aim import Run
from torchinfo import summary
from pathlib import PosixPath

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


def get_model(
    config_model: dict, device: torch.device
) -> Tuple[torch.nn.Module, Dataset]:
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
) -> DataLoader:
    if dataset_type == MapNetDataset and phase != "test":
        seq = config_dataloader.get("sequences", None)
        if seq:
            seq = seq[phase]
        dataset = MapNetDataset(
            path=dataset_path,
            steps=config_dataloader["step"],
            skip=config_dataloader["skip"],
            color_jitter=config_dataloader["color_jitter"],
            seq=seq,
            image_path=config_paths.get("images"),
            save_processed_dataset=config_dataloader.get(
                "save_processed_dataset", None
            ),
        )
    elif (
        dataset_type == MapNetDataset
        and phase == "test"
        and "sequences" in config_dataloader
    ):
        dataset = SevenScenes(
            dataset_path=PosixPath(dataset_path),
            seq=config_dataloader["sequences"][phase],
        )
    elif (
        dataset_type == MapNetDataset
        and phase == "test"
        and "images" in config_paths
    ):
        dataset = AbsolutePoseDataset(
            dataset_path=PosixPath(dataset_path),
            image_folder=PosixPath(config_paths["images"]),
            save_processed_dataset=config_paths.get(
                "save_processed_dataset", False
            ),
        )
    else:
        dataset = dataset_type(
            dataset_path=PosixPath(dataset_path),
            image_folder=PosixPath(config_paths["images"]),
            save_processed_dataset=config_paths.get(
                "save_processed_dataset", False
            ),
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
) -> Dict[str, DataLoader]:
    dataloaders = {}

    assert len(dataset_paths) == len(
        phases
    ), "Number of phases and dataset paths must be equal"

    for phase, dataset_path in zip(phases, dataset_paths):
        dataloaders[phase] = get_dataloader(
            dataset_path,
            config_dataloader,
            config_paths,
            dataset_type,
            phase,
        )
    return dataloaders


def get_optimizer(config_optimizer: dict, paramters) -> torch.optim.Optimizer:
    if config_optimizer["name"] == "SGD":
        optimizer = SGD(
            paramters,
            lr=config_optimizer["lr"],
            momentum=config_optimizer["momentum"],
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


def get_scheduler(
    config_scheduler: dict, optimizer: torch.optim.Optimizer
) -> Any:
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
        os.path.join(
            experiment_dir, config["environment"]["run_name"] + "_config.ini"
        ),
    )

    train_dataset_path = config["paths"]["train_dataset"]
    validation_dataset_path = config["paths"]["validation_dataset"]
    # test_dataset_path = config["paths"]["test_dataset"]

    model, dataset_type = get_model(config["model"], device)
    dataloaders = get_dataloaders(
        config["dataloader"],
        config["paths"],
        ["train", "validation"],
        [train_dataset_path, validation_dataset_path],
        dataset_type,
    )

    criterion = get_loss(config["loss"], device)

    # only parameters of final layer are being optimized
    if config["model"]["name"] == "posenet":
        optimizer = get_optimizer(config["optimizer"], model.parameters())
    elif config["model"]["name"] == "mapnet":
        param_list = list(model.parameters())
        if criterion.learn_beta:
            param_list.extend([criterion.sax, criterion.saq])
        if criterion.learn_gamma:
            param_list.extend([criterion.srx, criterion.srq])

        optimizer = get_optimizer(config["optimizer"], param_list)
    else:
        optimizer = get_optimizer(config["optimizer"], model.parameters())
    summary(
        model,
        (
            config["dataloader"]["batch_size"],
            *dataloaders["train"].dataset[0][0].size(),
        ),
    )
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
    ).cpu()
    net_weights_path = os.path.join(
        experiment_dir,
        config["environment"]["run_name"] + ".pth",
    )
    torch.save(trained_model.state_dict(), net_weights_path)


def test(config_path: str):
    config = ConfigParser(config_path)
    device = get_device()
    experiment_dir = create_experiment_dir(
        config["paths"]["net_weights_dir"],
        config["environment"]["experiment_name"],
    )
    train_configs = ConfigParser(
        os.path.join(
            experiment_dir, config["environment"]["run_name"] + "_config.ini"
        )
    )
    set_random_seed(train_configs["environment"]["seed"])

    dataset_path = config["paths"]["test_dataset"]

    model, dataset_type = get_model(train_configs["model"], device)
    weights_path = os.path.join(
        experiment_dir,
        config["environment"]["run_name"] + ".pth",
    )
    model.load_state_dict(torch.load(weights_path))
    model = model.to(get_device())

    if "sequences" in config["dataloader"]:
        train_configs["dataloader"]["sequences"] = config["dataloader"][
            "sequences"
        ]

    dataloaders = get_dataloaders(
        train_configs["dataloader"],
        config["paths"],
        ["test"],
        [dataset_path],
        dataset_type,
    )

    targets, predictions = test_model(model, dataloaders["test"], device)
    targets.to_csv(config["paths"]["targets"], index=False)
    predictions.to_csv(config["paths"]["predictions"], index=False)

    mae_xyz, mae_wxyz = calculate_MAE_poses(targets, predictions)
    print(f"MAE XYZ: {mae_xyz}")
    print(f"MAE WXYZ: {mae_wxyz}")


def inference(config_path="./inference.ini", image: Image = None):
    config = ConfigParser(config_path)
    device = get_device()
    experiment_dir = create_experiment_dir(
        config["paths"]["net_weights_dir"],
        config["environment"]["experiment_name"],
    )
    train_configs = ConfigParser(
        os.path.join(
            experiment_dir, config["environment"]["run_name"] + "_config.ini"
        )
    )
    set_random_seed(train_configs["environment"]["seed"])

    model, _ = get_model(train_configs["model"], device)
    weights_path = os.path.join(
        experiment_dir,
        config["environment"]["run_name"] + ".pth",
    )
    model.load_state_dict(torch.load(weights_path))
    model = model.to(get_device())
    model.eval()
    torch.set_grad_enabled(False)

    unit_measure = config["image_processing"]["unit_measure"]
    pixels_amount = config["image_processing"]["pixels_amount"]
    rotation_matrix = config["image_processing"]["rotation_matrix"]
    translation_vector = config["image_processing"]["translation_vector"]

    if image is not None and not isinstance(model, MeNet):
        transformers = get_image_transform()
        img = transformers(image).unsqueeze(0).unsqueeze(0).to(device)
        prediction = model(img)
        prediction = prediction.squeeze(0).squeeze(0)
        del model
        return (
            prediction.detach().cpu().numpy(),
            unit_measure,
            pixels_amount,
            rotation_matrix,
            translation_vector,
        )
    else:
        raise NotImplementedError("Inference mode not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base model")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Config file"
    )
    parser.add_argument(
        "-t", "--train", action="store_true", help="Train model flag"
    )
    parser.add_argument(
        "-i", "--inference", action="store_true", help="Inference model flag"
    )
    parser.add_argument(
        "-e", "--test", action="store_true", help="Test model flag"
    )

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
