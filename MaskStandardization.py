from utils import standardize_mask
import os

# Define the folder path containing the images
folder_path = 'datasets/masks'
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
    mask_files = os.listdir(cls_path)
    # create the folder
    new_cls_path = cls_path.replace('masks', 'standard_masks')
    os.makedirs(new_cls_path, exist_ok=True)
    count = 0

    for mask in mask_files:
        mask_path = os.path.join(folder_path, cls, mask)
        print('start to process image:' + mask_path)
        standardize_mask(mask_path)
        count += 1
        print('there are {} out of {} are finished'.format(count, len(mask_files)))





