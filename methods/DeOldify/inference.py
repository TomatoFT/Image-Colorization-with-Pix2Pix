import warnings

import fastai
from deoldify.visualize import *

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

colorizer = get_image_colorizer(artistic=True)

source_url = 'https://img.freepik.com/free-photo/grayscale-woman-headshot-serene-face-expression-portrait_53876-14465.jpg?w=900&t=st=1711731129~exp=1711731729~hmac=3474a14d05988097d83fc94d4fc86c248a6de2d64a0c4a235c9981ffaaf2ad87' #@param {type:"string"}
render_factor = 35  #@param {type: "slider", min: 7, max: 40}
watermarked = True #@param {type:"boolean"}

if source_url is not None and source_url !='':
    image_path = colorizer.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=True, watermarked=watermarked)
    show_image_in_notebook(image_path)
else:
    print('Provide an image url and try again.')