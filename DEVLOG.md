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


## 2026/07/24

### Progress
I modified the structure `GolfSwingDataset` to match the updated train/validation/test splitting process. Previously, the Dataset class received the root data directory, scanned each phase directory, and created the sample list internally.

Now, the splitting process creates `train_samples`, `val_samples`, and `test_samples` before the Dataset objects are initialized. Each sample is stored as a tuple containing the image's relative path and its numeric swing-phase label: `(image_path, numeric label)`

Since the sample lists already contain both the image path and label, `GolfSwingDataset` no longer needs to scan the raw dataset directories or maintain its own `class_to_idx` mapping. The class now only stores the provided samples, loads each image, applies the assigned transformation pipeline, and returns the transformed image with its label.

### Defining Train/Val/Test datasets
After completing the swing-level splitting process and defining the image transformation pipeline, I initialized the training, validation, and test Dataset objects.

The size of each dataset is:
- training: 72 images
- validation: 24 images
- test: 24 images

The current transformation pipeline resizes each image to `256 x 128` and converts it into a PyTorch tensor.

I also visually inspected the transformed training images using Matplotlib to confirm that the images were loaded correctly, resized to the expected dimensions, and matched their numeric labels.

The DataLoaders will be implemented next.


## 2026/07/27

### Progress
I finished implementing functions that create and return the Dataset and DataLoader objects. I added wrapper functions to make the data-preparation code cleaner, more modular, and easier to read.

The `create_datasets()` function creates and returns the training, validation, and test Dataset objects. The `create_dataloaders()` function then uses those Dataset objects to create the corresponding DataLoaders for model training, validation, and testing.

I am currently using a fixed random seed to make the train, validation and test splits reproducible. I may revise how the seed is handled later.

### Implementing the CNN Architecture
I completed the initial convolutional neural network architecture by defining a reusable `ConvBlock` class.

Each convolutional block contains:
- a `Conv2d` layer
- a ReLU activation function
- a 2D max-pooling layer

The main CNN uses three convolutional blocks with the following channel progression:
- 3 input channels -> 16 output channels
- 16 input channels -> 32 output channels
- 32 input channels -> 64 output channels

After the convolutional blocks, the model applies adaptive average pooling to produce feature maps with a fixed spatial size of 4 x 4. The output is then flattened and passed into a linear layer that produces four raw class scores, one for each golf swing phase.

### Implemented the Training Process
In `train.py`, I initialized:
- the number of epochs
- the training device
- the training, validation, and test DataLoaders
- the CNN model
- the cross-entropy loss function
- the Adam optimizer

I also implemented the training loop. For each epoch, the model processes all training batches, performs forward and backward propagation, and updates its parameters using the optimizer.

The training loop also calculates and reports the average training loss and training accuracy for each epoch.


## 2026/07/28

### Progress
I restructured the codebase to improve readability and organization, including adding wrapper functions and defining fixed configuration values as constants.

I also revised the code comments by removing unnecessary line-by-line explanations. Many comments were redundant because the corresponding code was already straightforward and self-explanatory.

In addition, I implemented the `train_one_epoch` and `validate` functions, which are executed during each epoch. Both functions return the model's average loss and accuracy for their respective datasets.

### Next Steps
The current model is ready for evaluation on the test dataset. BEfore concluding the project, I will decide whether additional experimentation is necessary.

Potential improvements include testing various hyperparameters, expanding the dataset, modifying the CNN architecture, and adding data augmentation. If further experimentation is not justified, I will evaluate the current model on the untouched test set, document the results and limitations, and conclude the project.