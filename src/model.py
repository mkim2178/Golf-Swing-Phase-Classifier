from torch import nn

class ConvBlock(nn.Module):
    """
    a convolutional block with including:
    - Conv2d (a 2 dimensional convolutional layer with a kernel size of 3, stride size 1, and 1 padding)
    - ReLU (a relu activation function with inplacement)
    - MaxPool2d (a 2 dimensional max pooling layer with a kernel size of 2 and stride of 2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    """
    a Convolutional Neural Network of the model
    since we have RGB images, the first input channel would be 3 and we have four different swing-phases, the output number of class would be 4
    - block1: a ConvBlock with input channels = 3, output channels = 16, kernel size of 3, stride size 1, and 1 padding
    - block2: a ConvBlock with input channels = 16, output channels = 32, kernel size of 3, stride size 1, and 1 padding
    - block3: a ConvBlock with input channels = 32, output channels = 64, kernel size of 3, stride size 1, and 1 padding
    - adaptive_pool: an Adaptive Average Pooling with using size of 4 (this manages a linear layer having a vast amount of features and occur overfitting)
    - flatten: a Flatten layer that converts current shape from [batch size, channel, height, width] to [batch size, channel * height * width]
    - linear: a Linear layer that returns each score of the swing phase
    """
    def __init__(self, in_channels=3, num_classes=4):
        super(CNN, self).__init__()

        # conv blocks
        self.block1 = ConvBlock(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # adaptive layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # linear layer
        self.linear = nn.Linear(in_features=64 * 4 * 4, out_features=num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x