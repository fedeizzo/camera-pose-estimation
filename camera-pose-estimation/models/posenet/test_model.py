import torch
import transforms3d.quaternions as quat
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
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


def from_relative_to_absolute_pose(
    predictions: pd.DataFrame
):
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
    model: torch.nn.Module, dataloaders: DataLoader
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    torch.set_grad_enabled(False)
    predictions = []
    targets = dataloaders.dataset.original_df
    for x, _ in dataloaders:
        predictions.append(model(x))

    if len(predictions):
        predictions = predictions[0]
    predictions = pd.DataFrame(
        predictions.cpu().data.numpy(),
        columns=["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
    )
    return targets, predictions
