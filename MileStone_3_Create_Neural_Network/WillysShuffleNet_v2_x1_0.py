from torch import nn
from torchvision import models
#from torchinfo import summary
#import torch

class myShuffleNet_v2_x1_0(nn.Module):
    def __init__(self, my_pretrained_model = models.shufflenet_v2_x0_5(pretrained=True),number_classes=18):
        super(myShuffleNet_v2_x1_0, self).__init__()
        num_ftrs = my_pretrained_model.fc.in_features
        my_pretrained_model.fc = nn.Linear(num_ftrs, out_features=number_classes, bias=True)
        self.myShuffleNet_v2_x1_0 = my_pretrained_model

    def forward(self, x):
        x = self.myShuffleNet_v2_x1_0(x)
        return nn.functional.log_softmax(x,dim=1)

#willyAlexNet = myShuffleNet_v2_x1_0()
#print(willyAlexNet)
