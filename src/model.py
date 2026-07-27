from torch import nn

class ConvBlock(nn.Module):
    """
    a Convolutional Block with a sequence of
    - Conv2d (convolutional 2d layer with kernel size of 3 x 3, stride of 1, and 1 padding)
    - ReLU (relu activation with inplacement)
    - MaxPool2d (max pooling 2d layer with kernel size of 2 x 2 and stride of 2)
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
    the actual Convolutional Neural Network of the model
    since we have RGB images, the firsts input channel would be 3 and the output number of class would be 4 (we have 4 swing phases)
    - block1: ConvBlock with input channels = 3, output channels = 16, kernel size of 3 x 3, stride as 1, and 1 padding
    - block2: ConvBlock with input channels = 16, output channels = 32, kernel size of 3 x 3, stride as 1, and 1 padding
    - block3: ConvBlock with input channels = 32, output channels = 64, kernel size of 3 x 3, stride as 1, and 1 padding
    - adaptive_pool: an Adaptive Average Pooling with using size of 4 x 4 (this will mitigate of having a vast amount of features on linear layer which can result an overfitting)
    - flatten: a flatting process from [batch size, channel, height, width] to [batch size, channel * height * width]
    - linear: a linear layer that returns each score of the swing phase
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