from torch import nn
from torchvision import models
#from torchinfo import summary
#import torch
#pretrained = models.alexnet(pretrained=True)

class MyAlexNet(nn.Module):
    def __init__(self, my_pretrained_model = models.alexnet(pretrained=True),number_classes=18):
        super(MyAlexNet, self).__init__()
        num_ftrs = my_pretrained_model.classifier[-1].in_features
        my_pretrained_model.classifier[-1] = nn.Linear(num_ftrs, out_features=number_classes, bias=True)
        self.WillyAlexNet = my_pretrained_model

    def forward(self, x):
        x = self.WillyAlexNet(x)
        return nn.functional.log_softmax(x,dim=1)
#willyAlexNet = MyAlexNet()
#print(willyAlexNet)
