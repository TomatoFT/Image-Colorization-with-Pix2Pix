# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Đọc ảnh từ file
# image = cv2.imread('/home/tomato/Desktop/Image-Colorization-with-Pix2Pix/experiments/generated/Pix2Pix/img34.png')

# # Load model Super-Resolution EDSR
# model_path = "EDSR_x4.pb"  # Đảm bảo rằng bạn đã tải mô hình từ OpenCV repository
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr.readModel(model_path)
# sr.setModel("edsr", 8)  # Sử dụng mô hình EDSR và tỷ lệ phóng to 4

# # Phóng to ảnh
# result = sr.upsample(image)

# # Hiển thị ảnh
# plt.figure(figsize=(10, 8))
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.title("Super-Resolution Image")
# plt.axis('off')
# plt.show()

from super_image import EdsrModel, ImageLoader
from PIL import Image

url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
image = Image.open("/home/tomato/Desktop/Image-Colorization-with-Pix2Pix/experiments/generated/Pix2Pix/img34.png")


def post_processing_image():
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)      # scale 2, 3 and 4 models available
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)

    ImageLoader.save_image(preds, './scaled_2x.png')                        # save the output 2x scaled image to `./scaled_2x.png`
    ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')     # save an output comparing the super-image with a bicubic scaling
