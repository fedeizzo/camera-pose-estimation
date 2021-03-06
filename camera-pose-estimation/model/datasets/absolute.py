import torch
import numpy as np
import pandas as pd
import os

# from dataset import load_images
from pathlib import PosixPath
from os import listdir
from os.path import isfile
from typing import List, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from transforms3d.quaternions import quat2mat, qnorm, qeye


def get_image_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_absolute_sample_from_row(df_row) -> Tuple[str, torch.Tensor]:
    """
    Extracts a sample from a dataset row
    """
    x = df_row.image
    y = torch.Tensor(
        [
            df_row.x,
            df_row.y,
            df_row.z,
            df_row.qx,
            df_row.qy,
            df_row.qz,
            df_row.qw,
        ]
    )

    return x, y


def homogeneous_to_quaternion(matrix) -> np.ndarray:
    """
    Transform homogeneous matrix to a quaternion representation
    (used only with 7scens dataset).
    """
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


def qlog(q: np.ndarray) -> np.ndarray:
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


def qlog_map(q) -> np.ndarray:
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def qexp_map(q) -> np.ndarray:
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
    return q


class AbsolutePoseDataset(Dataset):
    def __init__(
        self,
        dataset_path: PosixPath,
        image_folder: PosixPath,
        save_processed_dataset: bool = False,
    ) -> None:
        self.X = []
        self.Y = []
        if dataset_path.name.endswith(".csv"):
            df = pd.read_csv(dataset_path)

            if isinstance(df, pd.DataFrame):
                images = self.load_images(
                    image_folder,
                    list(df["image"].values),
                )

                for row in list(df.itertuples()):
                    curr_sample = get_absolute_sample_from_row(row)
                    # self.X.append(curr_sample[0])
                    self.Y.append(curr_sample[1])
            else:
                raise ValueError("Error loading dataset")
            self.X = torch.stack(images)
            self.Y = torch.stack(self.Y)
            if save_processed_dataset:
                processed_path = (
                    dataset_path.parent
                    / "processed_dataset"
                    / dataset_path.name.split(".")[0]
                )
                os.makedirs(processed_path, exist_ok=True)
                torch.save(
                    self.X,
                    processed_path / "dataset_X.pt",
                )
                torch.save(
                    self.Y,
                    processed_path / "dataset_Y.pt",
                )
        elif isfile(dataset_path / "dataset_X.pt") and isfile(
            dataset_path / "dataset_Y.pt"
        ):
            self.X = torch.load(dataset_path / "dataset_X.pt")
            self.Y = torch.load(dataset_path / "dataset_Y.pt")
        else:
            raise ValueError("Dataset format allowed are: csv and pt")

    def load_image(self, image_path: PosixPath) -> torch.Tensor:
        transforms = get_image_transform()
        return transforms(Image.open(image_path))

    def load_images(self, image_folder: PosixPath, image_names: List[str]):
        sorted_names = sorted(image_names)
        images = [
            self.load_image(os.path.join(image_folder, img)) for img in sorted_names
        ]
        return images

    def __getitem__(self, idxs):
        X = self.X[idxs]
        Y = self.Y[idxs]
        return X, Y

    def __len__(self):
        return len(self.X)


class SevenScenes(Dataset):
    def __init__(self, dataset_path: PosixPath, seq: str, use_qlog: bool = False):
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
        transforms = get_image_transform()
        return transforms(Image.open(image_path))

    def __getitem__(self, idxs):
        return self.X[idxs], self.Y[idxs]

    def __len__(self):
        return len(self.X)


class MapNetDataset(Dataset):
    def __init__(
        self,
        path: str,
        steps: int,
        skip: int,
        color_jitter: float,
        seq: Optional[str],
        image_path: Optional[str],
        save_processed_dataset: Optional[bool],
    ):
        if seq:
            self.inner_dataset = SevenScenes(PosixPath(path), seq)
        elif image_path is not None and save_processed_dataset is not None:
            self.inner_dataset = AbsolutePoseDataset(
                PosixPath(path), PosixPath(image_path), save_processed_dataset
            )
        else:
            raise ValueError(
                "current configuration cannot be used either with 7scenes or absolute pose datasets"
            )

        skips = skip * np.ones(steps - 1)
        skips = np.insert(skips, 0, 0)

        offsets = skips.cumsum()
        offsets -= offsets[int(len(offsets) / 2)]

        idxs = []
        for idx in range(len(self.inner_dataset)):
            tmp_idx = idx + offsets
            idxs.append(np.minimum(np.maximum(tmp_idx, 0), len(self.inner_dataset) - 1))

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
