from PIL import Image
# https://stable-diffusion-art.com/how-to-come-up-with-good-prompts-for-ai-image-generation/
from PIL import ImageFilter
# def convert_to_transparent(image_path, output_path):
#     # Load the image
#     image = Image.open(image_path)
#
#     # Convert the image to RGBA mode (with alpha channel)
#     image = image.convert("RGBA")
#
#     # Get the image dimensions
#     width, height = image.size
#
#     # Create a new transparent image with the same dimensions
#     transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#
#     # Iterate over each pixel and set the alpha channel based on the foreground color
#     for x in range(width):
#         for y in range(height):
#             r, g, b, a = image.getpixel((x, y))
#             if r < 10 and g < 10 and b < 10:
#                 # Set transparent for background pixels
#                 transparent_image.putpixel((x, y), (r, g, b, 0))
#             else:
#                 # Set opaque for foreground pixels
#                 transparent_image.putpixel((x, y), (r, g, b, 255))
#
#     # Save the transparent image as PNG
#     transparent_image.save(output_path, format="PNG")
#
#
#
# # Example usage
# input_image_path = "C:/Users/.../Downloads/segmented_image_new/n01689811/n01689811_17.JPEG"
# output_image_path = "C:/Users/.../Downloads/segmented_image_new/transparent_n01689811_17.png"
# convert_to_transparent(input_image_path, output_image_path)

from utils import convert_to_transparent_v2
import os
import logging
import cv2


logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Define the folder path containing the images
folder_path = 'datasets/image'
# Load the list of image file names in the folder
cls_folders = os.listdir(folder_path)
cls_count = 0
# Iterate over each image masks file
for cls in cls_folders:
    cls_count += 1
    print('-----------------------------------------')
    print('start to process cls {}, it is {} out of {}'.format(cls, cls_count, len(cls_folders)))
    print('-----------------------------------------')
    cls_path = os.path.join(folder_path, cls)
    img_files = os.listdir(cls_path)
    # create the folder
    new_cls_path = cls_path.replace('image', 'transparent_img')
    os.makedirs(new_cls_path, exist_ok=True)
    count = 0

    for img in img_files:
        image_path = os.path.join(folder_path, cls, img)
        print('start to process image:' + image_path)
        output_path = image_path.replace('JPEG','png')
        output_path = output_path.replace('image', 'transparent_img')
        try:
            convert_to_transparent_v2(image_path, output_path)
        except Exception as error:
            logging.error({'img': image_path, 'log': error})
        count += 1
        print('there are {} out of {} are finished'.format(count, len(img_files)))





