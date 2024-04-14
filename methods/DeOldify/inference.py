import warnings
import os
from deoldify.visualize import get_image_colorizer


def get_the_image_restoration(choice, path, output):
    warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
    image_path = ''
    colorizer = get_image_colorizer(artistic=True)
    if choice == "URL":
        render_factor = 35  #@param {type: "slider", min: 7, max: 40}
        watermarked = False #@param {type:"boolean"}

        if path is not None and path !='':
            image_path = colorizer.plot_transformed_image_from_url(url=path, 
                                                                   render_factor=render_factor, 
                                                                   compare=True)
        else:
            print('Provide an image url and try again.')
    elif choice == "Uploaded":
        render_factor = 35  #@param {type: "slider", min: 7, max: 40}
        watermarked = False #@param {type:"boolean"}

        if path is not None and path !='':
            image_path = colorizer.plot_transformed_image(path=path,
                                                          render_factor=render_factor,
                                                          results_dir=output,
                                                          compare=True)
        else:
            print('Provide an image url and try again.')

    else:
        print('INVALID CHOICE')

    print('You can see the result image in ', image_path)

if __name__ == "__main__":
    input_img_folder='/content/drive/MyDrive/experiment_dataset/input'
    real_img_folder='/content/drive/MyDrive/experiment_dataset/original'
    generated_img_folder='/content/drive/MyDrive/experiment_dataset/generated'
    

    input_images = os.listdir(input_img_folder)
    real_images = os.listdir(real_img_folder)
    # generated_images = os.listdir(generated_img_folder)

    for img in input_images:
        file = input_img_folder + '/' + img
        _ = get_the_image_restoration(choice="Uploaded", path=file, output=generated_img_folder)
        print("Done img ", img)

