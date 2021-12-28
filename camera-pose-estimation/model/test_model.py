import torch
import transforms3d.quaternions as quat
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from datasets.absolute import qexp_map
from models.mapnet import MapNet
from models.menet import MeNet
from typing import Tuple


def compute_absolute_positions(positions: pd.DataFrame) -> np.ndarray:
    tvecs = positions[["tx", "ty", "tz"]].values
    qvecs = positions[["qw", "qx", "qy", "qz"]].values
    xyz_positions = []

    for i in range(len(qvecs)):
        R = quat.quat2mat(qvecs[i])
        xyz_positions.append(np.dot(-(R.T), tvecs[i]))
    xyz_positions = np.array(xyz_positions, dtype=np.float32)

    return xyz_positions


def from_relative_to_absolute_pose(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.cumsum()


def reverse_normalization(
    positions: pd.DataFrame, quaternion_scaler, translation_scaler
) -> pd.DataFrame:
    for col in ["qx", "qy", "qz", "qw"]:
        updated = quaternion_scaler.inverse_transform(
            positions[col].values.reshape(-1, 1)
        ).flatten()
        positions.update({col: updated})

    for col in ["tx", "ty", "tz"]:
        updated = translation_scaler.transform(
            positions[col].values.reshape(-1, 1)
        ).flatten()
        positions.update({col: updated})

    return positions


def test_model(
    model: torch.nn.Module, dataloaders: DataLoader, device: torch.device
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    torch.set_grad_enabled(False)
    predictions = []
    targets = []
    for x, y in dataloaders:
        if isinstance(model, MapNet):
            x = torch.unsqueeze(x, dim=1).to(device=device)
        else:
            x = x.to(device=device)
        predictions.append(model(x))
        targets.append(y)

    if isinstance(model, MeNet):
        predictions = pd.DataFrame(predictions[0].numpy(), columns=['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz'])
        predictions = from_relative_to_absolute_pose(predictions)
        targets = pd.DataFrame(targets[0].numpy(), columns=['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz'])
        targets = from_relative_to_absolute_pose(targets)
    else:
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        predictions = predictions.squeeze(dim=1)

        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if predictions.shape[-1] == 6:
            predictions_quat = np.apply_along_axis(qexp_map, 1, predictions[:, 3:])
            predictions_xyz = predictions[:, :3]
            targets_quat = np.apply_along_axis(qexp_map, 1, targets[:, 3:])
            targets_xyz = targets[:, :3]

            predictions = np.concatenate([predictions_xyz, predictions_quat], axis=1)
            targets = np.concatenate([targets_xyz, targets_quat], axis=1)

        predictions = pd.DataFrame(
            predictions,
            columns=["tx", "ty", "tz", "qw", "qx", "qy", "qz"],
        )

        targets = pd.DataFrame(
            targets,
            columns=["tx", "ty", "tz", "qw", "qx", "qy", "qz"],
        )

    return targets, predictions
