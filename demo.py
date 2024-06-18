import sys
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append("./methods/Pix2Pix")
sys.path.append("./methods/GFPGAN")
sys.path.append("./methods/Deoldify")

from methods.Pix2Pix.config.constants import MEAN, STD
from methods.Pix2Pix.dataloader import Transform
from methods.Pix2Pix.train import load_model
from post_processing import post_processing_image, save_image

# Define the de-normalization function
def de_norm(img):
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach().numpy()
    img_ = np.transpose(img_, (1, 2, 0))
    return img_

# Define the post-processing sharpening function
def post_process_sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

# Load the generator model
generator = load_model('', demo=True)  # Replace with the path to your model weights
generator.eval()

def colorize_image(input_image, method):
    # Ensure the input image has the correct data type (uint8)
    img = Image.fromarray(np.array(input_image, dtype=np.uint8))
    
    # Process image with Pix2Pix method
    if method == 'Pix2Pix':
        original_size = img.size

        # Transform input image
        transformer = Transform()
        input_tensor = transformer(img) 
        input_tensor = torch.unsqueeze(input_tensor, 0)

        # Generate colorized image
        with torch.no_grad():
            colorized_tensor = generator(input_tensor)

        colorized_image = de_norm(colorized_tensor[0].cpu())

        # Clip the values and convert to uint8
        # colorized_image = np.clip(colorized_image, 0, 255).astype(np.uint8)

        # Convert numpy array to PIL Image and save it
        im = Image.fromarray((colorized_image * 1).astype(np.uint8)).convert('RGB')
        im.save("your_file.jpeg")
        image_name = save_image(colorized_image)

        result_img_path = post_processing_image(image=image_name)

        img = cv2.imread(result_img_path)
        return post_process_sharpen(cv2.resize(colorized_image, (original_size[0], original_size[1])))
    
    elif method == 'Deoldify':
        print('Input Image: ', input_image.name)
    else:
        raise ValueError("Invalid method selected. Please choose either 'Pix2Pix' or 'GFPGAN'.")

# Gradio Interface
iface = gr.Interface(
    fn=colorize_image,
    inputs=["image", gr.Dropdown(["Pix2Pix", "GFPGAN"], label="Method")],
    outputs="image",
    live=True,
    title="Image Colorization Demo",
    description="Upload a grayscale image to see it colorized using the GAN model.",
)

iface.launch(debug=True)
