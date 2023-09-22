import cv2
import numpy as np
import logging
import glob
import os
from PIL import Image


def convert_to_transparent(image_path, output_path):
    # Load the image
    image = Image.open(image_path)
    mask_path = image_path.replace('image', 'standard_masks')
    mask_path = mask_path.replace('JPEG', 'png')
    mask = Image.open(mask_path)

    # Convert the image to RGBA mode (with alpha channel)
    image = image.convert("RGBA")

    # Get the image dimensions
    width, height = image.size

    # Create a new transparent image with the same dimensions
    transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Iterate over each pixel and set the alpha channel based on the foreground color
    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))
            if mask.getpixel((x,y))==0:
                # Set transparent for background pixels
                transparent_image.putpixel((x, y), (r, g, b, 0))
            else:
                # Set opaque for foreground pixels
                transparent_image.putpixel((x, y), (r, g, b, 255))

    # Save the transparent image as PNG
    transparent_image.save(output_path, format="PNG")
    return


def convert_to_transparent_v2(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask_path = image_path.replace('image', 'standard_masks')
    mask_path = mask_path.replace('JPEG', 'png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Create a 4-channel image with alpha channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Iterate over each pixel and set the alpha channel based on the mask
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            alpha = mask[x, y]
            image[x, y, 3] = 0 if alpha == 0 else 255

    # Save the transparent image as PNG
    cv2.imwrite(output_path, image)



def standardize_mask(mask_path):
    '''
    standardize all masks to 0,255
    :param mask_path: the path of mask folder
    '''
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Create a new mask where non zero values are set to 255 and the value 0 remains unchanged
    new_mask = np.where(mask == 0, 0, 255)
    new_path = mask_path.replace('masks', 'standard_masks')
    cv2.imwrite(new_path, new_mask)
    return

def image_path_iterator(datasets_folder, file_type='jpg'):
    '''
    interator for image path
    :param datasets_folder: folder name for datasets
    :param file_type: file type, ex. jpg, png....
    :return: iterator for each class
    '''
    for root, _, _ in os.walk(datasets_folder):
        image_files = glob.glob(
            os.path.join(root, '*.'+file_type))  # Replace '*.jpg' with the appropriate image file extensions
        if image_files:
            yield image_files

# Replace 'path_to_datasets_folder' with the actual path to your "datasets" folder
datasets_folder = 'path_to_datasets_folder'
for image_path in image_path_iterator(datasets_folder):
    print(image_path)

def blur(image_path, mask_img_path, method=None):
    '''
    :param image_path:  path of the image path that need to be processed
    :param method:  None: apply blurring to all image;
                    backgroud: apply blurring to image background
                    object: apply blurring to image object
    :param mask_img_path: path of the mask of the image
    :return:  Nothing
    '''

    # use has_error to flag if there is issue
    has_error = False

    # Load the image
    image = cv2.imread(image_path)

    # Load the mask (assuming it's a grayscale image with values 0 and 255)
    try:
        mask = cv2.imread(mask_img_path, 0)
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    except cv2.error as error:
        logging.error({'mask': mask_img_path, 'img': image_path, 'log': error})
        return

    if method == 'background':
        # Invert the mask to select the background region
        # background_mask = cv2.bitwise_not(mask_3d)
        # Apply blur to the background region
        blurred_background = cv2.GaussianBlur(image, (15, 15), 3)
        try:
            blurred_image = np.where(mask_3d == 0, blurred_background, image)
        except ValueError as error:
            logging.error({'mask': mask_img_path, 'img': image_path, 'log': error})
            return

    elif method == 'object':
        # Apply blur to the object region
        blurred_object = cv2.GaussianBlur(image, (15, 15), 3)
        try:
            blurred_image = np.where(mask_3d > 0, blurred_object, image)
        except ValueError as error:
            logging.error({'mask': mask_img_path, 'img': image_path, 'log': error})
            return
    if not method:
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    # output image
    path = image_path.replace('image','blur_'+method)
    cv2.imwrite(path, blurred_image)
    print('image are saved to:' + path)
    return
