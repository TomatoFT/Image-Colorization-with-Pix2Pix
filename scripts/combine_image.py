import os

from PIL import Image

# Replace 'download_path' with the directory where you saved the downloaded images
download_path = '/kaggle/working/Image-Colorization-with-Pix2Pix/dataset/original'

# Replace 'output_path' with the directory where you want to save the combined images
output_path = '/kaggle/working/Image-Colorization-with-Pix2Pix/dataset/dataset'

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loop through each downloaded image
for i in range(1, 101):  # Assuming you downloaded 100 images
    original_image_path = os.path.join(download_path, f'image_{i}.jpg')
    bw_image_path = os.path.join('/kaggle/working/Image-Colorization-with-Pix2Pix/dataset/old', f'image_{i}.jpg')

    # Open the original and black and white images
    original_image = Image.open(original_image_path)
    bw_image = Image.open(bw_image_path)

    # Get the size of the original image
    width, height = original_image.size

    # Create a new image with double the width to combine them horizontally
    combined_image = Image.new('RGB', (width * 2, height))

    # Paste the original image on the left and the black and white image on the right
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(bw_image, (width, 0))

    # Save the combined image
    combined_image.save(os.path.join(output_path, f'image_{i}.jpg'))

    print(f"Combined and saved image {i}.")
