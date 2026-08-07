import random
from pathlib import Path
from torchvision import transforms
from dataset import GolfSwingDataset
from torch.utils.data import DataLoader

# controls reproducibility of the swing-level train/validation/test split
RANDOM_SEED = 42


# setting labels used to retrieve image files
SETTING_LABELS = [f"set{i:02d}" for i in range(1, 4)]

# swing-phase names and their corresponding numeric labels
PHASE_LABELS = [("address", 0), ("top", 1), ("impact", 2), ("finish", 3)]

# an image transformation pipeline for resizing and converting into a Tensor
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])


def split_swings():
    """
    shuffle the swing labels and split them into training, validation and test groups using a 6:2:2 ratio
    """
    
    swing_labels = [f"swing{i:02d}" for i in range(1, 11)]
    random.shuffle(swing_labels)

    # split the shuffled labels into training, validation, and test groups
    train_swings = swing_labels[:6]
    val_swings = swing_labels[6:8]
    test_swings = swing_labels[8:]

    return train_swings, val_swings, test_swings


def create_datasets():
    """
    create the training, validation, and test datasets
    """
    random.seed(RANDOM_SEED)

    # store (image path, numeric label) pairs for each dataset
    train_samples = []
    val_samples = []
    test_samples = []

    # create a separate swing-level split for each setting (set01, set02, set03)
    for setting in SETTING_LABELS:

        train_swings, val_swings, test_swings = split_swings()

        # add each phase image to the spilt assigned to its swing
        for phase_name, phase_label in PHASE_LABELS:

            # create the directory path for the current swing phase
            phase_dir = Path("data") / phase_name

            # iterate six swings for training dataset
            # create the image path and append its path-label pair
            for swing in train_swings:
                file_name = f"{setting}_{swing}_{phase_name}.PNG"
                image_path = phase_dir / file_name
                train_samples.append((image_path, phase_label))

            # same logic as the training dataset
            for swing in val_swings:
                file_name = f"{setting}_{swing}_{phase_name}.PNG"
                image_path = phase_dir / file_name
                val_samples.append((image_path, phase_label))

            # same logic as the training dataset
            for swing in test_swings:
                file_name = f"{setting}_{swing}_{phase_name}.PNG"
                image_path = phase_dir / file_name
                test_samples.append((image_path, phase_label))


    # create the datasets using their sample lists and the shared transform
    train_dataset = GolfSwingDataset(train_samples, TRANSFORM)
    val_dataset = GolfSwingDataset(val_samples, TRANSFORM)
    test_dataset = GolfSwingDataset(test_samples, TRANSFORM)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(batch_size=8):
    """
    create DataLoaders for the training, validation, and test datasets
    """

    train_dataset, val_dataset, test_dataset = create_datasets()

    # init train, validation, and test dataloaders that will be used into the actual model
    # use the provided batch size for all DataLoaders
    # shuffle only the training data so each epoch receives batches in a different order
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader