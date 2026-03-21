import os
import torch
from torchvision.utils import save_image


def save_generated_images(fake_images, epoch, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    images = fake_images[:16]
    images = (images + 1) / 2  # [-1,1] -> [0,1]

    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    save_image(images, save_path, nrow=4)
