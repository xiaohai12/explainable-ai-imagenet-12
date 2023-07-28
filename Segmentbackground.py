from PIL import Image
import numpy as np
import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def segment_objects(jpg_folder, coco_mask_folder, output_folder):
    create_folder_if_not_exists(output_folder)

    for file_name in os.listdir(jpg_folder):
        if file_name.endswith(".JPEG"):
            # Load the original image
            img = Image.open(os.path.join(jpg_folder, file_name))
            img_array = np.array(img)

            # Load the corresponding COCO format mask
            mask_file_name = file_name.replace(".JPEG", ".png")
            try:
                mask = Image.open(os.path.join(coco_mask_folder, mask_file_name))
                mask_array = np.array(mask)

                # Create a binary mask for the target object
                target_mask = np.zeros_like(mask_array)
                target_mask[mask_array == 255] = 1

                # Apply the binary mask to the original image
                #target_img = np.zeros_like(img_array)
                #for channel in range(3):  # Iterate over the RGB channels
                #    target_img[:, :, channel] = img_array[:, :, channel] * target_mask

                # Create an RGBA image for the target object
                target_img = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
                #target_img = Image.new("RGBA", (img_array.shape[0], img_array.shape[0]), (0, 0, 0, 0))
                for channel in range(3):  # Iterate over the RGB channels
                    target_img[:, :, channel] = img_array[:, :, channel]
                target_img[:, :, 3] = target_mask * 255  # Set the alpha channel based on the binary mask

                # Save the segmented image to the output folder
                output_path = os.path.join(output_folder, file_name.replace(".JPEG", "_segmented.png"))
                target_img_pil = Image.fromarray(target_img)
                target_img_pil.save(output_path)

            except Exception as e:
                print("Error processing mask for:", file_name)
                print("Error details:", e)


# Example usage
jpg_folder = "C:/Users/.../Downloads/image/n02992211"
coco_mask_folder = "C:/Users/.../Downloads/masks/n02992211"
output_folder = "C:/Users/.../Downloads/segmented_image_transparent/n02992211"

segment_objects(jpg_folder, coco_mask_folder, output_folder)

#segmenting image with binary mask image folder for each object within the folder


"""
import matplotlib.pyplot as plt

data = {9811: 1, 3334: 1, 5128: 7, 6245: 1, 5936: 1, 9022: 7, 7072: 0, 2877: 8, 4003: 4, 7058: 61, 2211: 64, 4500: 0}

# Sort the data by value in descending order
sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

# Get the x and y values for the bar plot
x_values = [str(x[0]) for x in sorted_data]
y_values = [x[1] for x in sorted_data]

# Set the figure size and create the bar plot
fig = plt.figure(figsize=(8, 6))
plt.bar(x_values, y_values, color='blue')

# Add labels and title to the plot
plt.xlabel('Image Folder ID')
plt.ylabel('Re-Annotation Images Amounts')
plt.title('Figure of Manual Annotator Re-Annotation Error Images on Different Folders')

# Add text labels to the bars
for i, v in enumerate(y_values):
    plt.text(i, v+2, str(v), ha='center', fontsize=8)

# Save the figure as a PNG file
#plt.savefig('Figure of Manual Annotator Re-Annotation Error Images on Different Folders.png', dpi=300)
"""

"""
import matplotlib.pyplot as plt

# Data to plot
data = [('Annotator A', '9811', 1), ('Annotator A', '3334', 1), ('Annotator A', '5128', 7),
        ('Annotator A', '6245', 1), ('Annotator B', '5936', 1), ('Annotator B', '9022', 7),
        ('Annotator C', '7072', 0), ('Annotator C', '2877', 8), ('Annotator C', '4003', 4),
        ('Annotator C', '7058', 61), ('Annotator C', '2211', 64), ('Annotator C', '4500', 0)]

# Prepare the data for plotting
annotators = sorted(list(set([d[0] for d in data])))
x_pos = range(len(annotators))
values = []
for annotator in annotators:
    values.append([d[2] for d in data if d[0] == annotator])

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, value in enumerate(values):
    ax.bar(x_pos[i], value, color=colors[i], label=annotators[i], alpha=0.8)

# Set the axis labels and title
ax.set_xlabel('Annotators')
ax.set_ylabel('Re-Annotation Images Amounts')
ax.set_title('Re-Annotation Images Amounts per annotator')

# Set the axis ticks and tick labels
ax.set_xticks(x_pos)
ax.set_xticklabels(annotators)

# Add a legend
ax.legend()

# Save the plot to a file
#plt.savefig('Re-Annotation Images Amounts per Annotator.png')

# Show the plot
#plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Define the data
data = [('Annotator A', '9811', 1), ('Annotator A', '3334', 1), ('Annotator A', '5128', 7),
        ('Annotator A', '6245', 1), ('Annotator B', '5936', 1), ('Annotator B', '9022', 7),
        ('Annotator C', '7072', 0), ('Annotator C', '2877', 8), ('Annotator C', '4003', 4),
        ('Annotator C', '7058', 61), ('Annotator C', '2211', 64), ('Annotator C', '4500', 0)]

# Group the data by annotator
annotators = {}
for annotator_id, class_id, num_reworked in data:
    if annotator_id not in annotators:
        annotators[annotator_id] = {'class_ids': [], 'num_reworked': []}
    annotator_data = annotators[annotator_id]
    annotator_data['class_ids'].append(class_id)
    annotator_data['num_reworked'].append(num_reworked)

# Create the bar chart for each annotator
num_annotators = len(annotators)
fig, axs = plt.subplots(1, num_annotators, figsize=(5*num_annotators, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # blue, orange, green
for i, (annotator_id, annotator_data) in enumerate(annotators.items()):
    class_ids = annotator_data['class_ids']
    num_reworked = annotator_data['num_reworked']
    ax = axs[i]
    x = np.arange(len(class_ids))
    ax.bar(x, num_reworked, color=colors[i%3])
    ax.set_xticks(x)
    ax.set_xticklabels(class_ids)
    ax.set_title(annotator_id)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Number of Reworked Images')

#plt.savefig('Re-Annotation Images tendency per Annotator.png')

"""



