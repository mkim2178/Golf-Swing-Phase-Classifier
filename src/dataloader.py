import random
from pathlib import Path
from torchvision import transforms
from dataset import GolfSwingDataset
import matplotlib.pyplot as plt

"""
For each setting (10 golf swings), split into a ratio of 6:2:2 (train:validation:test)
- train: 18 golf swings * 4 phases = 72 images
- validation: 6 golf swings * 4 phases = 24 images
- test: 6 golf swings * 4 phases = 24 images
total: 30 golf swings * 4 phases = 120 images
"""

# for reproducibility
random.seed(42)

# labels
setting_labels = ["set0" + str(i) for i in range(1, 4)]

phase_labels = [("address", 0), ("top", 1), ("impact", 2), ("finish", 3)]

# lists that will contain each train, validation, test swing phases
train_samples = []
val_samples = []
test_samples = []

def split_swings():

    swing_labels = ["swing0" + str(i) for i in range(1, 10)] + ["swing10"]
    
    # randomly shuffle the swing_labels list
    random.shuffle(swing_labels)

    # split the swing_labels into a ratio of 6:2:2 (train:val:test) for each setting of golf swing
    train_swings = swing_labels[:6]
    val_swings = swing_labels[6:8]
    test_swings = swing_labels[8:]

    return train_swings, val_swings, test_swings


# for each setting, call the spliting function
for setting in setting_labels:
    train_swings, val_swings, test_swings = split_swings()

    # for each train, val, and test swings, append each swing's phase image
    for phase in phase_labels:

        phase_name, phase_label = phase

        phase_dir = Path("data") / phase_name

        for swing in train_swings:
            file_name = setting + "_" + swing + "_" + phase_name + ".PNG"
            image_path = phase_dir / file_name
            train_samples.append((image_path, phase_label))
        
        for swing in val_swings:
            file_name = setting + "_" + swing + "_" + phase_name + ".PNG"
            image_path = phase_dir / file_name
            val_samples.append((image_path, phase_label))

        for swing in test_swings:
            file_name = setting + "_" + swing + "_" + phase_name + ".PNG"
            image_path = phase_dir / file_name
            test_samples.append((image_path, phase_label))


# define the pipeline of the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])


train_dataset = GolfSwingDataset(train_samples, transform)
val_dataset = GolfSwingDataset(val_samples, transform)
test_dataset = GolfSwingDataset(test_samples, transform)

for sample in train_dataset.samples:
    print(sample)