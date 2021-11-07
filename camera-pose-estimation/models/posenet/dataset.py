import os
import pandas as pd
import torch
import numpy as np

from enum import Enum
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class DatasetType(Enum):
    RELATIVE = 1
    ABSOLUTE = 2


def get_image_trasform(is_train=True):
    if is_train:
        resize_transform = T.Resize(256)
        crop_transform = T.RandomCrop(224)
    else:
        resize_transform = T.Resize(224)
        crop_transform = T.CenterCrop(224)

    transform = T.Compose(
        [
            resize_transform,
            crop_transform,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


def load_images(image_folder: str, images_names: set, transforms):
    return {
        i: transforms(Image.open(os.path.join(image_folder, i)))
        for i in images_names
    }


def get_relative_sample_from_row(df_row):
    """
    Helper function to retrieve one dataset sample from a single row of the pandas DataFrame
    """
    x = [df_row.image_t, df_row.image_t1]
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

        if transforms is None:
            transforms = get_image_trasform()

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


class RelativePoseDataset(Dataset):
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

        if transforms is None:
            transforms = get_image_trasform()

        if isinstance(df, pd.DataFrame):
            images_t = list(df["image_t"].values)
            images_t1 = list(df["image_t1"].values)
            image_names = set(images_t + images_t1)
            images = load_images(image_folder, image_names, transforms)
            for row in df.itertuples():
                curr_sample = get_relative_sample_from_row(row)
                self.X.append(curr_sample[0])
                self.Y.append(curr_sample[1])
        else:
            raise ValueError("Error loading dataset")

        self.X = self.X
        self.Y = torch.stack(self.Y)
        self.images = images
        self.device = device

    def __getitem__(self, idxs):
        X = [self.images[self.X[idxs][0]], self.images[self.X[idxs][1]]]
        return torch.cat(X, 0).to(self.device), self.Y[idxs].to(self.device)

    def __len__(self):
        return len(self.X)
