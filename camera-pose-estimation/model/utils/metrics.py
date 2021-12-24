import pandas as pd
from typing import Tuple
from sklearn.metrics import mean_absolute_error


def calculate_MAE_poses(
    gt_poses: pd.DataFrame, pred_poses: pd.DataFrame
) -> Tuple[float, float]:
    mae_xyz = mean_absolute_error(gt_poses.values[:, :3], pred_poses.values[:, :3])
    mae_wxyz = mean_absolute_error(gt_poses.values[:, 3:], pred_poses.values[:, 3:])

    return mae_xyz, mae_wxyz
