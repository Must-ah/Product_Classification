
from torch import nn
from torch.nn import functional as F
import torch
# device = torch.device('cuda:0')


def conv(inChannels, outChannels, dilation, groups, dropProbability=0, kernalSize=3):
    """
        Creates the convolution layer that consist of:
        conv2d, batchNorm2d and Mish activation function lastly also a dropout
    """
    return (
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=kernalSize, stride=1, dilation=dilation, groups=groups
                  ), nn.BatchNorm2d(outChannels), nn.Mish(inplace=True), nn.Dropout2d(dropProbability, inplace=True)
    )


def willy_block(inChannels, outChannels, dilation, groups, dropProbability=0, kernalSize=3):
    # return conv(inChannels,outChannels,dilation,groups,dropProbability=dropProbability,kernalSize=kernalSize)
    return nn.Sequential(*conv(inChannels, outChannels, dilation, groups, dropProbability=dropProbability, kernalSize=kernalSize))


def fullyConnected(inFeatures, outFeatures):
    """
        Creates the standard fully connected layer consist of:
        Linear, barchNorm1d and Mish activation functino
    """
    return (

        # nn.BatchNorm1d(num_features=inFeatures),
        nn.Linear(in_features=inFeatures, out_features=outFeatures),
        nn.BatchNorm1d(outFeatures),
        nn.Mish(inplace=True)
    )


def willy_fullyConnected_block(inFeatures, outFeatures):
    return nn.Sequential(*fullyConnected(inFeatures, outFeatures))


def willy_prediction_block(inFeatures, outFeatures):
    return nn.Sequential(
        nn.Linear(in_features=inFeatures, out_features=outFeatures),
        nn.BatchNorm1d(outFeatures),
        # nn.Mish(inplace=True)
        # F.log_softmax(outFeatures,dim=1)
    )


class WillyNet(nn.Module):
    """
    A CNN designed by Mustafa A-Hussein to predict what category the product is in an image.
    The goal of the design was to satisfy my curiosity and to gain experince in designing Neural Network Architectures
    I also wanted to explore how Mish activation function compares with others such like ReLU.
    Args:
        s (int): how many channel should the first conv2d layer have.
        ToDo add output (int): to decide how many classes you want.
        ToDo add desired shape i.e., how many convolution layer you want with their shape
        ToDo cont: decide how many fully connected layer you want.

    """

    def __init__(self, s=20) -> None:  # s = nr_channels_out
        super(WillyNet, self).__init__()

        # the first layer, takes in 3 channel input (RGB) image
        self.conv1 = willy_block(inChannels=3, outChannels=s*3,
                                 dilation=5, groups=3, dropProbability=0.0,
                                 kernalSize=20)

        self.conv2 = willy_block(inChannels=s*3, outChannels=s*3,
                                 dilation=3, groups=4, dropProbability=0.00,
                                 kernalSize=16)

        self.conv3 = willy_block(inChannels=s*3, outChannels=s*2,
                                 dilation=2, groups=2, dropProbability=0.00,
                                 kernalSize=14)

        self.conv4 = willy_block(inChannels=s*2, outChannels=s*1,
                                 dilation=1, groups=1, dropProbability=0.00,
                                 kernalSize=12)

        self.conv5 = willy_block(inChannels=s*1, outChannels=s,
                                 dilation=1, groups=1, dropProbability=0,
                                 kernalSize=8)

        # This part I did do manually until I was satisfied with the architecture of the model
        self.flatten = nn.Flatten()
        self.fullyConnectedLayer = willy_fullyConnected_block(
            inFeatures=3920, outFeatures=512)
        self.predictionLayer = willy_prediction_block(
            inFeatures=512, outFeatures=18)

    def forward(self, x):
        # Convolution part
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # fully connected part
        x = self.flatten(x)
        x = self.fullyConnectedLayer(x)
        x = self.predictionLayer(x)
        return F.log_softmax(x, dim=1)


# myNet = WillyNet()
# from torchinfo import summary
# summary(myNet)
