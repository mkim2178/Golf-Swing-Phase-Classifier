from PIL import Image
from torch.utils.data import Dataset

class GolfSwingDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: a list of tuples (Path object, swing-phase label)
        transform: a transformation pipeline that will be applied to images
        """

        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        """
        return the length of "samples" list
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        get a certain image's path and label from samples list, convert into RGB image, and apply transformation if it's assigned
        return the image and the numeric label
        """
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
        return img, label