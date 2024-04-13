import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.utils import save_image
from tqdm import tqdm

from config.constants import DEVICE, MEAN, RESIZE, STD

preprocess = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def de_norm(img):
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach().numpy()
    img_ = np.transpose(img_, (1, 2, 0))
    return img_


def evaluate(val_dl, name, G):
    with torch.no_grad():
        real_images, generated_images = [], []
        print(val_dl)
        for input_img, real_img in tqdm(val_dl):
            input_img = input_img.to(DEVICE)
            real_img = real_img.to(DEVICE)

            fake_img = G(input_img)
            
            batch_size = input_img.size()[0]
            batch_size_2 = batch_size * 2

            for i in range(batch_size):
                real_images.append(torch.tensor(de_norm(real_img[i])))
                generated_images.append(torch.tensor(de_norm(fake_img[i])))

        real_images = torch.stack(real_images)
        generated_images = torch.stack(generated_images)

        _, axes = plt.subplots(6, 5, figsize=(8, 12))
        ax = axes.ravel()

    output_dir_original = "../../experiments/original/Pix2Pix/"
    output_dir_generated = "../../experiments/generated/Pix2Pix/"

    # Create directories if they don't exist
    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_generated, exist_ok=True)

    for input_img, real_img in tqdm(val_dl):
        input_img = input_img.to(DEVICE)
        real_img = real_img.to(DEVICE)

        fake_img = G(input_img)
        batch_size = input_img.size()[0]
        batch_size_2 = batch_size * 2

        for i in range(batch_size):
            ax[i].imshow(input_img[i].permute(1, 2, 0))
            ax[i + batch_size].imshow(de_norm(real_img[i]))
            ax[i + batch_size_2].imshow(de_norm(fake_img[i]))

            # Save original and generated images
            original_img_path = os.path.join(output_dir_original, f"img_{i * batch_size + i + 1}.png")
            generated_img_path = os.path.join(output_dir_generated, f"img_{i * batch_size + i + 1}.png")

            # Convert tensors to PIL images and save
            original_img = Image.fromarray((de_norm(real_img[i]) * 255).astype(np.uint8))
            generated_img = Image.fromarray((de_norm(fake_img[i]) * 255).astype(np.uint8))

            original_img.save(original_img_path)
            generated_img.save(generated_img_path)

            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i + batch_size].set_xticks([])
            ax[i + batch_size].set_yticks([])
            ax[i + batch_size_2].set_xticks([])
            ax[i + batch_size_2].set_yticks([])

            if i == 0:
                ax[i].set_ylabel("Input Image", c="g")
                ax[i + batch_size].set_ylabel("Real Image", c="g")
                ax[i + batch_size_2].set_ylabel("Generated Image", c="r")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("results/latest_results.jpg")
        break

# Add this utility function for matrix square root
def sqrtm(mat, eps=1e-10):
    eigval, eigvec = np.linalg.eigh(mat)
    # Ensure non-negative eigenvalues (add epsilon for stability)
    eigval[eigval < 0] = eps
    
    # Sort eigenvalues in descending order and corresponding eigenvectors
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    
    # Take the square root of the diagonal matrix of eigenvalues
    sqrt_diag = np.diag(np.sqrt(np.maximum(eigval, eps)))
    
    # Calculate the square root of the covariance matrix
    sqrt_cov = eigvec @ sqrt_diag @ np.linalg.inv(eigvec)
    
    return sqrt_cov.real



# Add this utility function for covariance calculation
def torch_cov(m, rowvar=False):
    if rowvar:
        m = m.transpose(-1, -2)
    else:
        m = m.transpose(-2, -3)
    m_shape = m.size()
    m = m.reshape(m_shape[0], m_shape[1], -1)
    fact = 1.0 / (m.size(-1) - 1)
    m -= torch.mean(m, dim=-1, keepdim=True)
    mt = m.transpose(-1, -2)
    return fact * m.matmul(mt).squeeze()