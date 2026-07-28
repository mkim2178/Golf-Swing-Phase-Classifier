import random
from pathlib import Path
from torchvision import transforms
from dataset import GolfSwingDataset
from torch.utils.data import DataLoader

# reproducibility purpose
RANDOM_SEED = 42


# setting labels will be used to retrieve image files
SETTING_LABELS = ["set0" + str(i) for i in range(1, 4)]

# labels for each swing-phase (can be replaced with using a dictionary)
PHASE_LABELS = [("address", 0), ("top", 1), ("impact", 2), ("finish", 3)]

# a pipeline for the image transformation (resize and convert the image into a Tensor object)
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])


def split_swings():
    """
    shuffle the order of swing labels and split them into a ratio of 6:2:2 (train:val:test) for each setting of the golf swing
    """

    # "swing10" starts with "swing1" so it's has been separately added
    swing_labels = ["swing0" + str(i) for i in range(1, 10)] + ["swing10"]
    
    random.shuffle(swing_labels)

    # list slice the shuffled swing_labels to assign train, val, and test list
    train_swings = swing_labels[:6]
    val_swings = swing_labels[6:8]
    test_swings = swing_labels[8:]

    return train_swings, val_swings, test_swings


def create_datasets():
    """
    create datasets for train, val, and test
    """
    random.seed(RANDOM_SEED)

    # init lists that will contain a tuple (image path, numeric label) for each train, val, and test
    train_samples = []
    val_samples = []
    test_samples = []

    # for each setting labels (set0, set1, set2)
    for setting in SETTING_LABELS:

        train_swings, val_swings, test_swings = split_swings()

        # for each train, val, and test swings, append each swing's phase image
        for phase in PHASE_LABELS:

            # each phase has the name and the assigned numeric label (ex: (address, 0))
            phase_name, phase_label = phase

            # init a Path object with using the root directory and the phase name 
            phase_dir = Path("data") / phase_name

            # iterate six swings for training dataset
            for swing in train_swings:

                # init the name of image file, extend the Path object with the current image, and append it into the train_samples list as a tuple (image path, numeric swing-phase label)
                file_name = setting + "_" + swing + "_" + phase_name + ".PNG"
                image_path = phase_dir / file_name
                train_samples.append((image_path, phase_label))

            # same logic as the training dataset
            for swing in val_swings:
                file_name = setting + "_" + swing + "_" + phase_name + ".PNG"
                image_path = phase_dir / file_name
                val_samples.append((image_path, phase_label))

            # same logic as the training dataset
            for swing in test_swings:
                file_name = setting + "_" + swing + "_" + phase_name + ".PNG"
                image_path = phase_dir / file_name
                test_samples.append((image_path, phase_label))


    # init train, validation, and test datasets with applying each samples list and the transformation pipeline that has been initialized above
    train_dataset = GolfSwingDataset(train_samples, TRANSFORM)
    val_dataset = GolfSwingDataset(val_samples, TRANSFORM)
    test_dataset = GolfSwingDataset(test_samples, TRANSFORM)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(batch_size=8):
    """
    create dataloaders for train, val, and test
    """

    # init datasets
    train_dataset, val_dataset, test_dataset = create_datasets()

    # init train, validation, and test dataloaders that will be used into the actual model
    # batch size is 8 for all dataloaders
    # shuffle only the training data so each epoch receives batches in a different order
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader