import torch
import numpy as np
import pandas as pd
import os

# from dataset import load_images
from pathlib import PosixPath
from os import listdir
from typing import Optional

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from transforms3d.quaternions import quat2mat, qnorm, qeye


def get_absolute_sample_from_row(df_row):
    """
    Helper function to retrieve one dataset sample from a single row of the pandas DataFrame
    """
    x = df_row.image
    y = torch.Tensor(
        [
            df_row.tx,
            df_row.ty,
            df_row.tz,
            df_row.qx,
            df_row.qy,
            df_row.qz,
            df_row.qw,
        ]
    )

    return x, y


def homogeneous_to_quaternion(matrix):
    q = np.empty((4,), dtype=float)
    M = np.array(matrix, dtype=float, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / np.sqrt(t * M[3, 3])
    return q


def qlog(q: np.ndarray):
    q = np.array(q)  # To ensure there is a dtype
    qnorm_ = qnorm(q)
    if qnorm_ == 0.0:
        return qeye(q.dtype)

    w, v = q[0], q[1:]
    vnorm = np.sqrt(np.dot(v, v))
    result = np.zeros((4,), q.dtype)

    if vnorm == 0.0:
        return qeye(q.dtype)

    result[0] = np.log(qnorm_)
    result[1:] = v / vnorm * np.arccos(w / qnorm_)
    return result


def qlog_map(q):
  """
  Applies logarithm map to q
  :param q: (4,)
  :return: (3,)
  """
  if all(q[1:] == 0):
    q = np.zeros(3)
  else:
    q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
  return q

def qexp_map(q):
  """
  Applies the exponential map to q
  :param q: (3,)
  :return: (4,)
  """
  n = np.linalg.norm(q)
  q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
  return q


class AbsolutePoseDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        image_folder: str,
        device,
        is_train=True,
        transforms=None,
    ) -> None:
        self.X = []
        self.Y = []
        df = pd.read_csv(dataset_path)

        transforms = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
            ]
        )

        if isinstance(df, pd.DataFrame):
            images = load_images(
                image_folder, set(list(df["image_t"].values)), transforms
            )
            for row in df.itertuples():
                curr_sample = get_absolute_sample_from_row(row)
                self.X.append(curr_sample[0])
                self.Y.append(curr_sample[1])
        else:
            raise ValueError("Error loading dataset")

        self.X = self.X
        self.Y = torch.stack(self.Y)
        self.images = images
        self.device = device

    def __getitem__(self, idxs):
        X = torch.Tensor(self.images[self.X[idxs]]).to(self.device)
        Y = self.Y[idxs].to(self.device)
        return X, Y

    def __len__(self):
        return len(self.X)


class SevenScenes(Dataset):
    def __init__(
        self, dataset_path: PosixPath, seq: str, use_qlog: bool = True
    ):
        self.use_qlog = use_qlog
        sequence_path = os.path.join(dataset_path, seq)
        files = listdir(sequence_path)

        image_paths = list(
            map(
                lambda x: os.path.join(sequence_path, x),
                list(filter(lambda x: True if "color" in x else False, files)),
            )
        )
        image_paths.sort()
        pose_paths = list(
            map(
                lambda x: os.path.join(sequence_path, x),
                list(filter(lambda x: True if "pose" in x else False, files)),
            )
        )
        pose_paths.sort()

        self.X = torch.stack(
            list(map(lambda image: self.load_image(image), image_paths))
        )
        self.Y = torch.Tensor(
            np.array(list(map(lambda pose: self.load_pose(pose), pose_paths)))
        )

    def load_pose(self, pose_path: PosixPath) -> np.ndarray:
        homogeneous_matrix = np.loadtxt(pose_path)
        translation_vector = homogeneous_matrix[:3, 3]
        quaternion = np.array((homogeneous_to_quaternion(homogeneous_matrix)))

        # quat_normalized = quaternion / np.sqrt(np.dot(quaternion, quaternion))
        # quat_log = qlog(quat_normalized)
        quat_log_mapped = qlog_map(quaternion)
        rotation_matrix = quat2mat(quaternion)
        xyz_position = np.dot(-(rotation_matrix.T), translation_vector)
        if self.use_qlog:
            return np.concatenate((xyz_position, quat_log_mapped))
        else:
            return np.concatenate((xyz_position, quaternion))

    def load_image(self, image_path: PosixPath) -> torch.Tensor:
        transforms = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transforms(Image.open(image_path))

    def __getitem__(self, idxs):
        return self.X[idxs], self.Y[idxs]

    def __len__(self):
        return len(self.X)


class MapNetDataset(Dataset):
    def __init__(
        self,
        path: PosixPath,
        steps: int,
        skip: int,
        color_jitter: float,
        seq: Optional[str],
    ):
        if seq:
            self.inner_dataset = SevenScenes(path, seq)
        else:
            raise NotImplementedError(
                "Support for other datasets not implemented"
            )

        skips = skip * np.ones(steps - 1)
        skips = np.insert(skips, 0, 0)

        offsets = skips.cumsum()
        offsets -= offsets[int(len(offsets) / 2)]

        idxs = []
        for idx in range(len(self.inner_dataset)):
            tmp_idx = idx + offsets
            idxs.append(
                np.minimum(np.maximum(tmp_idx, 0), len(self.inner_dataset) - 1)
            )

        self.X = torch.from_numpy(np.array(idxs)).long()
        self.transforms = T.Compose(
            [
                T.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=0.5,
                )
            ]
        )

    def __getitem__(self, idxs):
        images, poses = self.inner_dataset[self.X[idxs]]
        # images = self.transforms(images)

        return (images, poses)

    def __len__(self):
        return self.X.shape[0]
