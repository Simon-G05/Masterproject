"""Functions to create a dataset for AnomalyCLIP."""
import numpy as np
from PIL import Image
import os
from pathlib import Path

import torchvision.transforms as transforms
from transform import image_transform
from constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def saveHeatmap(pathOut: str,
                heatmap: list,
                name: str,
                fileType: str = "png"
                ):
    if heatmap.dtype != np.float32 and heatmap.dtype != np.float64:
        raise ValueError("Heatmap must be float32 or float64")

    if not ((0 <= heatmap).all() and (heatmap <= 1).all()):
        raise ValueError("Heatmap values must be in the range [0, 1]")
            
    pathFile = Path(pathOut) / f"{name}.{fileType}"
    
    # Skaliere auf 16-Bit
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    os.makedirs(pathOut, exist_ok=True)
    # Speichern als .tif
    img = Image.fromarray(heatmap_uint8)
    img.save(pathFile)

    return str(pathFile)

# def apply_ad_scoremap(image, scoremap, alpha=0.5):
#     np_image = np.asarray(image, dtype=float)
#     scoremap = (scoremap * 255).astype(np.uint8)
#     scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
#     scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
#     return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    return preprocess, target_transform