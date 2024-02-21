import os

import requests

from config import ROOT_PATH

# Replace 'download_path' with the directory where you want to save the images
download_path = ROOT_PATH + 'dataset/original'

# Create the download directory if it doesn't exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Specify the number of images to download
num_images = 100

# Lorem Picsum API endpoint for random photos
url = f'https://picsum.photos/v2/list?page=1&limit={num_images}'

# Make a request to the Lorem Picsum API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Download each image
    for i, photo in enumerate(data):
        photo_url = f'https://picsum.photos/id/{photo["id"]}/{photo["width"]}/{photo["height"]}'
        image_response = requests.get(photo_url)

        # Save the image to the download directory
        with open(os.path.join(download_path, f'image_{i+1}.jpg'), 'wb') as f:
            f.write(image_response.content)

        print(f"Downloaded image {i+1}/{num_images}")

else:
    print(f"Failed to fetch images. Status code: {response.status_code}")
