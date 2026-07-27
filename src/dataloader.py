import random
from pathlib import Path
from torchvision import transforms
from dataset import GolfSwingDataset
from torch.utils.data import DataLoader

"""
For each setting (10 golf swings), split into a ratio of 6:2:2 (train:validation:test)
- train: 18 golf swings * 4 phases = 72 images
- validation: 6 golf swings * 4 phases = 24 images
- test: 6 golf swings * 4 phases = 24 images
total: 30 golf swings * 4 phases = 120 images
"""

# for reproducibility
RANDOM_SEED = 42

# labels
SETTING_LABELS = ["set0" + str(i) for i in range(1, 4)]
PHASE_LABELS = [("address", 0), ("top", 1), ("impact", 2), ("finish", 3)]

# define the pipeline of the image transformation
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])

def split_swings():

    swing_labels = ["swing0" + str(i) for i in range(1, 10)] + ["swing10"]
    
    # randomly shuffle the swing_labels list
    random.shuffle(swing_labels)

    # split the swing_labels into a ratio of 6:2:2 (train:val:test) for each setting of golf swing
    train_swings = swing_labels[:6]
    val_swings = swing_labels[6:8]
    test_swings = swing_labels[8:]

    return train_swings, val_swings, test_swings


def create_datasets():
    random.seed(RANDOM_SEED)

    # lists that will contain each train, validation, test swing phases
    train_samples = []
    val_samples = []
    test_samples = []

    # for each setting, call the spliting function
    for setting in SETTING_LABELS:
        train_swings, val_swings, test_swings = split_swings()

        # for each train, val, and test swings, append each swing's phase image
        for phase in PHASE_LABELS:

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

    # initialize train, validation, and test datasets
    train_dataset = GolfSwingDataset(train_samples, TRANSFORM)
    val_dataset = GolfSwingDataset(val_samples, TRANSFORM)
    test_dataset = GolfSwingDataset(test_samples, TRANSFORM)

    # # check the length of each dataset (expected values: 72, 24, 24)
    # print("---Length of Dataset---")
    # print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(batch_size=8):

    # init datasets
    train_dataset, val_dataset, test_dataset = create_datasets()

    # initialize train, validation, and test dataloaders that will be used into the actual model
    # batch size will be 8 for all of dataloaders and shuffle only the training data so each epoch receives batches in a different order
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # # check the length of batches for each dataloader (expected values: 9, 3, 3)
    # print("---Length of Batch---")
    # print(f"Train: {len(train_dataloader)}, Validation: {len(val_dataloader)}, Test: {len(test_dataloader)}")

    # # check the shape of a single batch from te train_dataloader
    # images, labels = next(iter(train_dataloader))
    # print(images.shape) # this gives [8, 3, 256, 128] = [batch size, channels, height, width]
    # print(labels.shape) # this gives [8] = one label for each image (8 images)

    return train_dataloader, val_dataloader, test_dataloader