import cv2
import numpy as np
import cv2
import os
"""
Funtion: segment_images
input should be v7 lab  semantic segmentation image mask, 
which means the image foreground should be the red color and background should be the black color
output should be the segmented image with the same name as the original image

for this code you cannot use the binary mask  to segment the image, 
which means the gray image generated from the labelme cannot be used by this code

"""
def segment_images(original_folder, mask_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image files in the original folder
    original_files = [f for f in os.listdir(original_folder) if f.endswith('.JPEG')]

    for original_file in original_files:
        # Load the original image
        original_path = os.path.join(original_folder, original_file)
        img = cv2.imread(original_path)

        # Get the corresponding mask file path
        mask_file = original_file.replace('.JPEG', '.png')
        mask_path = os.path.join(mask_folder, mask_file)

        # Check if the mask file exists
        if os.path.exists(mask_path):
            # Load the mask image
            mask = cv2.imread(mask_path)

            # Convert the mask image to grayscale
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Apply color thresholding to create a binary mask
            _, thresholded = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

            # Apply the binary mask to the original image
            segmented_image = cv2.bitwise_and(img, img, mask=thresholded)

            # Save the segmented image to the output folder
            output_path = os.path.join(output_folder, original_file)
            cv2.imwrite(output_path, segmented_image)

            print(f"Segmented image saved: {output_path}")
        else:
            print(f"No mask file found for: {original_file}")

# Example usage
original_folder = 'C:/Users/.../Downloads/image/n02992211'
mask_folder = 'C:/Users/.../Downloads/n02992211/masks'
output_folder = 'C:/Users/.../Downloads/segmented_image/segmented_image/n02992211'

segment_images(original_folder, mask_folder, output_folder)



# Display the segmented image
#cv2.imshow('Segmented Image', segmented_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()