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
from numpy import ndarray
import uuid
from datetime import datetime

def load_pretrained_model():
    return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)    

model = load_pretrained_model()

def save_image(image: ndarray) -> str:
    name = uuid.uuid3(namespace=uuid.NAMESPACE_X500, name=str(datetime.now()))
    ImageLoader.save_image(image, f'./demo_images/inputs/{name}.jpg')
    return name + '.jpg'

def post_processing_image(image: str) -> str:
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)

    ImageLoader.save_image(inputs, f'./demo_images/inputs/{image}')
    ImageLoader.save_image(preds, f'./demo_images/outputs/{image}')
    ImageLoader.save_compare(inputs, preds, f'./demo_images/compared/{image}')

    return image
