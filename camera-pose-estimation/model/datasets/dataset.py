import os

from PIL import Image
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


def load_images(image_folder: str, images_names: set, transforms):
    return {
        i: transforms(Image.open(os.path.join(image_folder, i))) for i in images_names
    }
