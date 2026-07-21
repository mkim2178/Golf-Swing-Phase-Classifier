from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

class GolfSwingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: a Path object of a root directory
        transform: a transformation pipeline that will be applied into images
        class_to_idx: a dictionary that maps the class name into an integer label
        samples: a list of tuples (image path, class index)
        """

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = {
            "address": 0,
            "top": 1,
            "impact": 2,
            "finish": 3
        }
        self.samples = []
        for cls, idx in self.class_to_idx.items():
            # create a path to the current class directory
            class_dir = self.data_dir / cls

            # append every image path and index by tuple
            for img_path in sorted(class_dir.glob("*.PNG")):
                self.samples.append((img_path, idx))
    
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