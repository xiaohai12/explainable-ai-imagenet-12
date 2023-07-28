#https://learnopencv.com/ai-art-generation-tools/
#the foreground image is resized to match the size of the background image.
# The resized foreground image is then blended with the background image using the cv2.addWeighted() function,
# which performs an element-wise addition of the two images with equal weights.
# This will give you a blended result.

import cv2
import os
import random

# Set paths to the foreground segmented images and unrelated background images folders
foreground_folder = "C:/Users/.../Downloads/segmented_image/segmented_image/n01689811"
background_folder = "C:/Users/.../Downloads/ImageNet-13 Original Image Dateset/image/n02123159"
output_folder = "C:/Users/.../Downloads/combine/n01689811/"

# Get the list of foreground and background images
foreground_images = os.listdir(foreground_folder)
background_images = os.listdir(background_folder)

# Shuffle the lists to ensure randomness
random.shuffle(foreground_images)
random.shuffle(background_images)

# Determine the number of images to generate
num_images = min(len(foreground_images), len(background_images))


# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Iterate over the number of images to generate
for i in range(num_images):
    foreground_image_name = foreground_images[i]
    background_image_name = background_images[i]

    foreground_image_path = os.path.join(foreground_folder, foreground_image_name)
    background_image_path = os.path.join(background_folder, background_image_name)

    foreground_image = cv2.imread(foreground_image_path)
    background_image = cv2.imread(background_image_path)

    # Resize the foreground image to match the background image size
    foreground_resized = cv2.resize(foreground_image, (background_image.shape[1], background_image.shape[0]))

    # Blend the foreground and background images
    combined_image = cv2.addWeighted(background_image, 1, foreground_resized, 1, 0)



    # Get the extension of the foreground image
    extension = foreground_image_name.split(".")[-1]

    # Save the resulting combined image with the same name and format as the foreground image
    combined_image_name = foreground_image_name
    combined_image_path = os.path.join(output_folder, combined_image_name)
    cv2.imwrite(combined_image_path, combined_image)





'''
# Iterate over the background images
for background_image_name in os.listdir(background_folder):
    background_image_path = os.path.join(background_folder, background_image_name)
    background_image = cv2.imread(background_image_path)

    # Iterate over the foreground segmented images
    for foreground_image_name in os.listdir(foreground_folder):
        if not foreground_image_name.endswith(".JPEG"):
            continue

        foreground_image_path = os.path.join(foreground_folder, foreground_image_name)
        foreground_image = cv2.imread(foreground_image_path)

        # Resize the foreground image to match the background image size
        foreground_resized = cv2.resize(foreground_image, (background_image.shape[1], background_image.shape[0]))

        # Blend the foreground and background images
        combined_image = cv2.addWeighted(background_image, 1, foreground_resized, 1, 0)

        # Save the resulting combined image to the output folder
        combined_image_path = os.path.join(output_folder, foreground_image_name)
        cv2.imwrite(combined_image_path, combined_image)


        foreground_image_path = os.path.join(foreground_folder, foreground_image_name)
        foreground_image = cv2.imread(foreground_image_path)

        # Randomly choose a location to place the foreground on the background
        x = random.randint(0, background_image.shape[1] - foreground_image.shape[1])
        y = random.randint(0, background_image.shape[0] - foreground_image.shape[0])

        # Create a mask of the foreground object
        foreground_mask = (foreground_image[:, :, 0] != 0) | (foreground_image[:, :, 1] != 0) | (foreground_image[:, :, 2] != 0)

        # Blend the foreground and background images
        combined_image = background_image.copy()
        combined_image[y:y + foreground_image.shape[0], x:x + foreground_image.shape[1]][foreground_mask] = foreground_image[foreground_mask]

        # Save the resulting combined image to the output folder
        combined_image_path = os.path.join(output_folder, foreground_image_name)
        cv2.imwrite(combined_image_path, combined_image)

'''






