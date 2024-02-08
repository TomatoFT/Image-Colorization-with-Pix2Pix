import os

from PIL import Image

# Replace 'download_path' with the directory where you saved the downloaded images
download_path = '/content/drive/MyDrive/dataset_imgClr/original'

# Replace 'output_path' with the directory where you want to save the transformed images
output_path = '/content/drive/MyDrive/dataset_imgClr/old'

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Function to convert an image to black and white
def convert_to_bw(image_path, output_path):
    image = Image.open(image_path).convert('L')  # 'L' mode for grayscale
    image.save(output_path)

# Loop through each downloaded image
for i in range(1, 101):  # Assuming you downloaded 100 images
    image_path = os.path.join(download_path, f'image_{i}.jpg')
    output_image_path = os.path.join(output_path, f'image_{i}.jpg')

    convert_to_bw(image_path, output_image_path)

    print(f"Transformed image {i} to black and white.")
