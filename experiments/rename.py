import os
import re


def rename_files_in_directory(directory):
    # Get a list of all items in the directory
    items = os.listdir(directory)

    # Iterate over each item in the directory
    for item in items:
        # Construct the full path to the item
        item_path = os.path.join(directory, item)

        # Check if the item is a file
        if os.path.isfile(item_path):
            # Extract the file name and extension
            filename, extension = os.path.splitext(item)

            # Remove underscores from the file name using regular expressions
            new_filename = re.sub('_', '', filename)

            # Construct the new file name with the original extension
            new_item_path = os.path.join(directory, new_filename + extension)

            # Rename the file
            os.rename(item_path, new_item_path)
            print(f"Renamed {item} to {new_filename + extension}")

if __name__ == "__main__":
    # Specify the directory containing the files to be renamed
    directory_path = 'original/Pix2Pix'

    # Call the function to rename files in the directory
    rename_files_in_directory(directory_path)
