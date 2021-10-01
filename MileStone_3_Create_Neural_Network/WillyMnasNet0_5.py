from torch import nn
from torchvision import models
#from torchinfo import summary
#import torch
#pretrained = models.mnasnet0_5(pretrained=True)

class MyMnasNet0_5(nn.Module):
    def __init__(self, my_pretrained_model=models.mnasnet0_5(pretrained=True)):
        super(MyMnasNet0_5, self).__init__()
        self.pretrained = my_pretrained_model
        self.nr_ftrs = self.pretrained.classifier[1].in_features
        self.pretrained.classifier[1] = nn.Linear(self.nr_ftrs, out_features=18, bias=True)
    def forward(self, x):
        x = self.pretrained(x)
        return nn.functional.log_softmax(x,dim=1)

#willyMnasnet0_5 = MyMnasNet0_5(pretrained)
#summary(willyMnasnet0_5)
