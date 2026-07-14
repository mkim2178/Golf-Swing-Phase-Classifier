"""
This is currently a testing step to understand loading the raw image and transform (convert) it into a Tensor.
Nothing special here, just for a testing purpose.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# image path for a test image file
image_path = Path("data/address/test.png")

# resize the image and convert into a Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# image & tensor image
image = Image.open(image_path)
image_tensor = transform(image)

# print out the size of each image to check if it's converted or note
print("original image size:", image.size)
print("tensor image size", image_tensor.shape)

# change the dimension of the image (this is only for displaying the image with using matplotlib) [C, H, W] -> [H, W, C]
display_image = image_tensor.permute(1, 2, 0)

# display the image
plt.imshow(display_image)
plt.axis("off")
plt.show()