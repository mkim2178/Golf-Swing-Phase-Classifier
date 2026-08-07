from PIL import Image
from torch.utils.data import Dataset


class GolfSwingDataset(Dataset):

    def __init__(self, samples, transform=None):
        """
        samples: a list of tuples (Path object, numeric label for each swing-phase)
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
        - load the image path & the numeric label at the given index
        - image is converted to RGB and transformed if a transformation pipeline if provided
        """
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
                
        return img, label