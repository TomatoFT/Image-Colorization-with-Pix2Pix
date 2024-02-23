import os
import shutil

from config import ROOT_PATH
from sklearn.model_selection import train_test_split

# Replace 'combined_path' with the directory where you saved the combined images
combined_path = ROOT_PATH + 'dataset/dataset'

# Replace 'output_path' with the directory where you want to save the train, val, and test sets
output_path = ROOT_PATH + 'dataset/dataset'

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# List all combined images in the 'combined' folder
combined_images = [f for f in os.listdir(combined_path) if f.endswith('.jpg')]

# Split the images into train, val, and test sets
train_images, remaining_images = train_test_split(combined_images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(remaining_images, test_size=0.5, random_state=42)

# Function to move images to their respective folders
def move_images(image_list, destination_path):
    for image in image_list:
        source = os.path.join(combined_path, image)
        destination = os.path.join(destination_path, image)
        shutil.move(source, destination)

# Create train, val, and test folders
train_path = os.path.join(output_path, 'train')
val_path = os.path.join(output_path, 'val')
test_path = os.path.join(output_path, 'test')

for folder in [train_path, val_path, test_path]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Move images to their respective folders
move_images(train_images, train_path)
move_images(val_images, val_path)
move_images(test_images, test_path)

print("Images have been moved to train, val, and test folders.")
