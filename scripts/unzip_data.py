import os
from zipfile import ZipFile


def unzip_all_files(folder_path):
    # Ensure the folder path ends with a slash
    folder_path = os.path.join(folder_path, '')

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file in files:
        # Check if the file is a zip file
        if file.endswith('.zip'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file)

            # Create a ZipFile object
            with ZipFile(file_path, 'r') as zip_ref:
                # Extract all contents to the folder
                zip_ref.extractall(folder_path)

            print(f"Unzipped {file} successfully.")

# Example usage
folder_to_unzip = 'YOUR ZIP FILE DATA PATH HERE'
unzip_all_files(folder_to_unzip)
