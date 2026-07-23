# Development Log

## 2026/07/13

### Progress
I set up the project environment and recorded ten golf swings at an outdoor driving range. However, I decided to record those swings again because I forgot to use slow-motion mode. Without slow-motion footage, it was difficult to capture clear images of the backswing top and impact phases because those moments occur very quickly.

I also gained a small blister on my left thumb while recording.

### Collecting Dataset
Because the weather was too hot, I decided to record the remaining twenty golf swings indoor after re-recording the ten outdor swings.

For the indoor swings, I plan to wear different outfits to increase the visual diversity of the datase. I'm not sure how much this will improve the model's generalization, but it should reduce the change that the model reies too heavily on one specific outfit.

### Environment Setup
I encountered several compatibility problems while installing PyTorch on my local machine.

I use an Intel-based MacBook Pro, and recent PyTorch versions no longer provide prebuilt packages for Intel macOS. Therefore, I created a Python 3.11 environment and installed versions compatible with PyTorch 2.2.2.

Current environment:
- python: 3.11.1
- numpy: 1.26.4
- pytorch: 2.2.2
- torchvision: 0.17.2


## 2026/07/14

### Progress
I re-recorded ten golf swings using slow-motion video. From each recording, I captured one image for each swing phase: address, top, impact, and finish. I then stored the images in their corresponding class directories.

This resulted in 40 images:
- 10 address images
- 10 backswing-top images
- 10 impact images
- 10 finish images

### Natural Variation in the Dataset
Because each swing was recorded separately, the extracted images are slightly different. The exact frame selected for each phase varies sightly between swings. The outdoor background also changes because of differences in sunlight, cloud movement, and the wind.

These small variations may help prevent the CNN from memorizing one identical image or background.


## 2026/07/21

### Progress
I recorded ten additional golf swings in slow-motion at the same outdoor driving range I visited on 07/14. The swing-phase capture process remained the same, but I wore a different outfit, and the rainy weather naturally introduced variation in the lighting and background.

I now have 20 recorded swings, with one imgae for each of following phases:
- Address
- Top
- Impact
- Finish

This gives me a total of 80 images.

### Implemented a Custom Dataset
I implemented a `GolfSwingDataset` class that inherits from PyTorch's built-in `Dataset` class.
- `__init__`: Initializes the dataset's root directory (`data_dir`), an optional transformation pipeline (`transform`), a dictionary that maps each swing-phase to a numeric label (`class_to_idx`), and a list of `(image path, class index)` tuples (`samples`). The current mapping is:
    - Address: 0
    - Top: 1
    - Impact: 2
    - Finish: 3
- `__len__`: Returns the number of entries in `samples`, which represents the total number of images in the dataset.
- `__getitem__`: Takes an index, retrieves the corresponding image path and label from `samples`, opens the image, converts it to RGB, and applies the transformation pipeline if one was provided. It then returns the image and the numeric label.


### Removed `transform.py`
I removed `transform.py` because it was only used as a temporary testing area for inspecting raw images and experimenting with resizing and tensor conversion. These transformations will now be defined outside the dataset class and applied dynamically inside `GolfSwingDataset.__getitem__`.


## 2026/07/23

### Progress
I recorded the final ten golf swings on 2026/07/22. They were recorded at the same location, but the background was different because I recorded them at night, and the weather was slightly rainy. I now have all 30 golf swings, or 120 images, that will be used as the dataset for my model.

### Implemented the Train/Validation/Test Splitting process and Image transformation pipeline
I manually implemented the dataset splitting process without using built-in dataset splitting functions. I decided to use 72 images for training, 24 images for validaion, and 24 images for  testing. 

The dataset is split by complete swing, meaning that the address, top, impact, and finish images from the same swing remain in the same subset. Each of the three recording settings contributes 6 swings to the training set, 2 swings to the validation set, and 2 swings to the test set. This keeps the subsets balanced across the different backgrounds and outfits while preventing data leakge between them.

### Transformation Pipeline
The image transformation pipeline was implemented in the same file as the splitting process. The pipeline is relatively simple because it only resizes each image and converts it into a PyTorch tensor.

The actual training, validation, and test dataset objects, along with their corresponding DataLoaders, will be implemented next.