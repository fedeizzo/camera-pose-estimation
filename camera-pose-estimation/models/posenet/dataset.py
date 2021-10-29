import os
import pandas as pd
import torch

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

    transform = T.Compose([
        resize_transform,
        crop_transform,
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform


def get_sample_from_row(df_row, image_folder, transforms):
    """
    Helper function to retrieve one dataset sample from a single row of the pandas DataFrame
    """
    img = Image.open(os.path.join(image_folder, df_row.image))
    x = transforms(img)
    y = torch.Tensor([
        df_row.tx,
        df_row.ty,
        df_row.tz,
        df_row.qx,
        df_row.qy,
        df_row.qz,
        df_row.qw,
    ])

    return x, y


class HighMemoryDataset(Dataset):
    def __init__(self, dataset_path: str, image_folder: str, is_train=True, transforms=None) -> None:
        self.X = []
        self.Y = []
        df = pd.read_csv(dataset_path)

        if transforms is None:
            transforms = get_image_trasform(is_train)

        for row in df.itertuples():
            curr_sample = get_sample_from_row(
                row,
                image_folder,
                transforms
            )
            self.X.append(curr_sample[0])
            self.Y.append(curr_sample[1])
        
    def __getitem__(self, idxs):
        return self.X[idxs], self.Y[idxs]

    def __len__(self):
        return len(self.X)


class LowMemoryDataset(Dataset):
    def __init__(self, dataset_path: str, image_folder: str, is_train=True, transforms=None) -> None:
        self.transforms = transforms
        self.image_folder = image_folder
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
                    row,
                    self.image_folder,
                    self.transforms
                )
                x.append(curr_sample[0])
                y.append(curr_sample[1])

            x = torch.stack(x)            
            y = torch.stack(y)
        else:
            x, y = self._get_sample(sample)

        return x, y

    def __len__(self):
        return len(self.df)
