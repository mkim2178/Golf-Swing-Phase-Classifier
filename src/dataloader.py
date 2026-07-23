import random
from torchvision import transforms
from dataset import GolfSwingDataset

# for reproducibility
random.seed(42)

# lists that will contain each train, validation, test swing phases
train = []
val = []
test = []


def train_val_test_split():
    """
    a function that randomly shuffles and splits swing labels into a ratio of 6:2:2 (train:val:test) for each swing setting
    """

    # 10 swings for each setting
    swing_labels = ["swing01_",
                    "swing02_",
                    "swing03_",
                    "swing04_",
                    "swing05_",
                    "swing06_",
                    "swing07_",
                    "swing08_",
                    "swing09_",
                    "swing10_",
                    ]
    
    # randomly shuffle the swing_labels list
    random.shuffle(swing_labels)

    # list slicing for each train, validation, and test swings
    train_swings = swing_labels[:6]
    val_swings = swing_labels[6:8]
    test_swings = swing_labels[8:]

    return train_swings, val_swings, test_swings


# for each setting, call the spliting function
for setting in ["set01_", "set02_", "set03_"]:
    train_swings, val_swings, test_swings = train_val_test_split()

    # for each train, val, and test swings, append each swing's phase image
    for phase in ["address", "top", "impact", "finish"]:
        train += [setting + swing + phase + ".PNG" for swing in train_swings]
        val += [setting + swing + phase + ".PNG" for swing in val_swings]
        test += [setting + swing + phase + ".PNG"for swing in test_swings]

"""
for each setting (10 golf swings), split into a ratio of 6:2:2 (train:validation:test)
- train: 18 golf swings * 4 phases = 72 images
- validation: 6 golf swings * 4 phases = 24 images
- test: 6 golf swings * 4 phases = 24 images
total: 30 golf swings * 4 phases = 120 images
"""
for t in train:
    print(t)
print(f"Number of images for train: {len(train)}")
print(f"Number of images for validation: {len(val)}")
print(f"Number of images for test: {len(test)}")




# define the pipeline of the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor()
])

# WORK IN PROGRESS (NEED TO DEFINE THE DATALOADER AND THE ACTUAL DATASET THAT WILL BE USED IN THE MODEL)