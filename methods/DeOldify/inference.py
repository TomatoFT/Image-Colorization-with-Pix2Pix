import warnings
import os
from deoldify.visualize import get_image_colorizer


def get_the_image_restoration(choice, path):
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
            image_path = colorizer.plot_transformed_image(url=path,
                                                          render_factor=render_factor, 
                                                          compare=True)
        else:
            print('Provide an image url and try again.')

    else:
        print('INVALID CHOICE')

    print('You can see the result image in ', image_path)

if __name__ == "__main__":
    input_img_folder = ""
    real_img_folder = ""
    generated_img_folder = ""


    input_images = os.listdir(input_img_folder)
    real_images = os.listdir(real_img_folder)
    # generated_images = os.listdir(generated_img_folder)

