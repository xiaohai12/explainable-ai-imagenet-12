
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
# Load the PNG mask image
mask_image = np.array(PIL.Image.open('C:/Users/.../Downloads/masks/n02607072/n02607072_14589.png'), dtype=np.int32)

# Display the mask image
plt.imshow(mask_image)
plt.show()










