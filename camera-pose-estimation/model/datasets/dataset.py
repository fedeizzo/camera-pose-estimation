import os

from PIL import Image

def load_images(image_folder: str, images_names: set, transforms):
    return {
        i: transforms(Image.open(os.path.join(image_folder, i))) for i in images_names
    }
