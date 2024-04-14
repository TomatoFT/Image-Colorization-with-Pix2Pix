import glob
import os

import cv2
import numpy as np
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from matplotlib import pyplot as plt
from PIL import Image


def restore_the_image_gfpgan(
                            input_path: str, 
                            output_path: str,
                            version: int=1,
                            upscale: int=2,
                            bg_upsampler: str= 'realesrgan',
                            bg_tile :int=400,
                            suffix:bool=None,
                            ext:str='auto',
                            weight:str = '',
                            aligned:bool = True,
                            only_center_face:bool = False
                            ):
    # ------------------------ input_path & output_path ------------------------
    if input_path.endswith('/'):
        input_path = input_path[:-1]
    if os.path.isfile(input_path):
        img_list = [input_path]
    else:
        img_list = sorted(glob.glob(os.path.join(input_path, '*')))

    os.makedirs(output_path, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=False)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        print(input_img)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(output_path, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(output_path, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(output_path, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if ext == 'auto':
                extension = ext[1:]
            else:
                extension = ext

            if suffix is not None:
                save_restore_path = os.path.join(output_path, 'restored_imgs', f'{basename}_{suffix}.{extension}')
            else:
                save_restore_path = os.path.join(output_path, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{output_path}] folder.')


def evaluate(input_img_folder,
             real_img_folder, 
             generated_img_folder, 
             G):
    
    input_images = os.listdir(input_img_folder)
    real_images = os.listdir(real_img_folder)
    generated_images = os.listdir(generated_img_folder)
    
    with torch.no_grad():
        restore_the_image_gfpgan(input_path=input_img_folder, 
                                 output_path=generated_img_folder)

        _, axes = plt.subplots(6, 5, figsize=(8, 12))
        ax = axes.ravel()

    # Create directories if they don't exist
    os.makedirs(generated_img_folder, exist_ok=True)
    os.makedirs(real_img_folder, exist_ok=True)

    for input_img, real_img, generated_img in zip(input_images, real_images, generated_images):
        fake_img = generated_img
        batch_size = input_img.size()[0]
        batch_size_2 = batch_size * 2

        for i in range(batch_size):
            ax[i].imshow(input_img[i].permute(1, 2, 0))
            ax[i + batch_size].imshow(real_img[i])
            ax[i + batch_size_2].imshow(generated_img[i])

            # Save original and generated images
            original_img_path = os.path.join(real_img_folder, f"img_{i * batch_size + i + 1}.png")
            generated_img_path = os.path.join(generated_img_folder, f"img_{i * batch_size + i + 1}.png")

            # Convert tensors to PIL images and save
            original_img = Image.fromarray(real_img[i] * 255).astype(np.uint8)
            generated_img = Image.fromarray(generated_img[i] * 255).astype(np.uint8)

            original_img.save(original_img_path)
            generated_img.save(generated_img_path)

            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i + batch_size].set_xticks([])
            ax[i + batch_size].set_yticks([])
            ax[i + batch_size_2].set_xticks([])
            ax[i + batch_size_2].set_yticks([])

            if i == 0:
                ax[i].set_ylabel("Input Image", c="g")
                ax[i + batch_size].set_ylabel("Real Image", c="g")
                ax[i + batch_size_2].set_ylabel("Generated Image", c="r")

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("results/latest_results.jpg")
        break

if __name__ == "__main__":
    evaluate(input_img_folder='',
             real_img_folder=''
             )