import sys

sys.path.append("./methods/Pix2Pix")
sys.path.append("./methods/GFPGAN")
sys.path.append("./methods/Deoldify")

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from methods.Pix2Pix.config.constants import MEAN, STD
from methods.Pix2Pix.dataloader import Transform
from methods.Pix2Pix.train import load_model
from methods.DeOldify.inference import get_the_image_restoration

# Define the Generator and Discriminator

def de_norm(img):
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach().numpy()
    img_ = np.transpose(img_, (1, 2, 0))
    return img_


def post_process_sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


# Load your Generator model weights
generator = load_model('45', demo=True)  # Replace with the path to your model weights
generator.eval()

def colorize_image(input_image, method):
    # Convert Grayscale to RGB
    transformer = Transform()
    
    # Ensure the input image has the correct data type (uint8)
    img = Image.fromarray(np.array(input_image, dtype=np.uint8))
        # Generate colorized image based on selected method
    if method == 'Pix2Pix':
        original_size = img.size

        # Transform input image
        input_tensor = transformer(img)
        input_tensor = torch.unsqueeze(input_tensor, 0)

        # Generate colorized image
        with torch.no_grad():
            colorized_tensor = generator(input_tensor)

        colorized_image = de_norm(colorized_tensor[0].cpu())

        return post_process_sharpen(cv2.resize(colorized_image, (original_size[0], original_size[1])))
    
    elif method == 'GFPGAN':
        # _ = get_the_image_restoration(choice="Uploaded", path=file, output=generated_img_folder)
        pass
    else:
        raise ValueError("Invalid method selected. Please choose either 'Pix2Pix' or 'GFPGAN'.")

# Gradio Interface
iface = gr.Interface(
    fn=colorize_image,
    inputs=["image", gr.inputs.Dropdown(["Pix2Pix", "GFPGAN"], label="Method")],
    outputs="image",
    live=True,
    title="Image Colorization Demo",
    description="Upload a grayscale image to see it colorized using the GAN model.",
)

iface.launch()
