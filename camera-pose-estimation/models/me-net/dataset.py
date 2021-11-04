import os
import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


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


def load_images(image_folder: str, images_names: np.ndarray, transforms, device):
    return {
        i: transforms(Image.open(os.path.join(image_folder, i))).to(device)
        for i in images_names
    }


def get_sample_from_row(df_row):
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


class HighMemoryDataset(Dataset):
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
            transforms = get_image_trasform(is_train)

        if isinstance(df, pd.DataFrame):
            images = load_images(image_folder, df["image_t"].values, transforms, device)
            for row in df.itertuples():
                curr_sample = get_sample_from_row(row)
                self.X.append(curr_sample[0])
                self.Y.append(curr_sample[1])
        else:
            raise ValueError("Error loading dataset")

        self.X = self.X
        self.Y = torch.stack(self.Y).to(device)
        self.images = images
        self.device = device

    def __getitem__(self, idxs):
        X = [self.images[self.X[idxs][0], self.X[idxs][0]]]
        return torch.stack(X).to(self.device), self.Y[idxs]

    def __len__(self):
        return len(self.X)


class LowMemoryDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        image_folder: str,
        device,
        is_train=True,
        transforms=None,
    ) -> None:
        self.transforms = transforms
        self.image_folder = image_folder
        self.device = device
        self.df = pd.read_csv(dataset_path)

        if transforms is None:
            self.transforms = get_image_trasform(is_train)

    def __getitem__(self, idxs):
        sample = self.df.iloc[idxs]

        # check if data from multiple indexes has been requested
        if isinstance(sample, pd.DataFrame):
            x, y = [], []
            for row in sample.itertuples():
                curr_sample = get_sample_from_row(
                    row, self.image_folder, self.transforms
                )
                x.append(curr_sample[0])
                y.append(curr_sample[1])

            x = torch.stack(x).to(self.device)
            y = torch.stack(y).to(self.device)
        else:
            x, y = self._get_sample(sample)

        return x, y

    def __len__(self):
        return len(self.df)
