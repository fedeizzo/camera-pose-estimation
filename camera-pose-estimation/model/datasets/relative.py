import torch
import pandas as pd

from torchvision import transforms as T
from .dataset import load_images
from typing import Tuple
from torch.utils.data import Dataset

def get_image_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def get_relative_sample_from_row(df_row) -> Tuple[list, torch.Tensor]:
    """
    Extracts a sample from a dataset row
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


class RelativePoseDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        image_folder: str,
        save_processed_dataset: bool = False,
    ) -> None:
        self.X = []
        self.Y = []
        df = pd.read_csv(dataset_path)

        transforms = get_image_transform()

        if isinstance(df, pd.DataFrame):
            self.original_df = df.copy(deep=True)
            df.drop(index=0, inplace=True)
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

    def __getitem__(self, idxs):
        X = [self.images[self.X[idxs][0]], self.images[self.X[idxs][1]]]
        return torch.cat(X, 0), self.Y[idxs]

    def __len__(self):
        return len(self.X)
