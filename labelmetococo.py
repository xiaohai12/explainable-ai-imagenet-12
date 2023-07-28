import os
import cv2
import numpy as np

def segment_images(mask_path, images_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if the mask image exists
    #if not os.path.isfile(mask_path):
    #    print("Mask image does not exist.")
    #    return

    # Load the binary mask image
    #mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Iterate over each image file in the images folder
    for image_file in os.listdir(mask_path):
        # Load the image
        print(image_file)
        #dir, file = os.path.split(json_path)
        file_name = image_file.split('.')[0]

        originalimage = os.path.join(images_folder, file_name + '.JPEG')
        print(originalimage)
        mask_image = cv2.imread(os.path.join(mask_path, image_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(originalimage)

        # Apply the mask to the original image
        segmented_image = cv2.bitwise_and(image, image, mask=mask_image)

        # Save the segmented image
        output_path = os.path.join(output_folder, file_name + '.JPEG')
        cv2.imwrite(output_path, segmented_image)

        print("Segmented image saved:", output_path)


# Example usage
mask_path = "C:/Users/.../Downloads/image/n02992211/mask/n02992211/"
images_folder = "C:/Users/.../Downloads/image/n02992211"
output_folder = "C:/Users/.../Downloads/segmented_image/segmented_image/n02992211"
segment_images(mask_path, images_folder, output_folder)
