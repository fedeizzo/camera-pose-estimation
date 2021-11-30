import torch
import transforms3d.quaternions as quat
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from datasets.absolute import qexp_map
from typing import Tuple


def compute_absolute_positions(positions: pd.DataFrame):
    tvecs = positions[["tx", "ty", "tz"]].values
    qvecs = positions[["qw", "qx", "qy", "qz"]].values
    xyz_positions = []

    for i in range(len(qvecs)):
        R = quat.quat2mat(qvecs[i])
        xyz_positions.append(np.dot(-(R.T), tvecs[i]))
    xyz_positions = np.array(xyz_positions, dtype=np.float32)

    return xyz_positions


def from_relative_to_absolute_pose(predictions: pd.DataFrame):
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
        x = torch.unsqueeze(x, dim=1).to(device=device)
        predictions.append(model(x))
        targets.append(y)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    predictions = predictions.squeeze(dim=1)

    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    predictions_quat = np.apply_along_axis(qexp_map, 1, predictions[:, 3:])
    predictions_xyz = predictions[:, :3]
    targets_quat = np.apply_along_axis(qexp_map, 1, targets[:, 3:])
    targets_xyz = targets[:, :3]

    predictions = pd.DataFrame(
        np.concatenate((predictions_xyz, predictions_quat), 1),
        columns=["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )

    targets = pd.DataFrame(
        np.concatenate((targets_xyz, targets_quat), 1),
        columns=["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )

    return targets, predictions
