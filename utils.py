import cv2
import numpy as np
import logging


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