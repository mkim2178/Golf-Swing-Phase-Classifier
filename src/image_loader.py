from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

image_path = Path("data/address/test.png")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path)
image_tensor = transform(image)

print("original image size:", image.size)
print("tensor image size", image_tensor.shape)

display_image = image_tensor.permute(1, 2, 0)

plt.imshow(display_image)
plt.axis("off")
plt.show()