import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from utils  import *


class model_resnet18(nn.Module):
    def __init__(self,num_classes):
        super(model_resnet18,self).__init__()

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        output = self.fc(feature)
        return feature,output

class model_resnet50(nn.Module):
    def __init__(self,num_classes):
        super(model_resnet50,self).__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        output = self.fc(feature)
        return feature,output



# for temporal convolution flating layer
class TemporalLayer(nn.Module):
    def __init__(self):
        super(TemporalLayer,self).__init__()
        # kernal = [[0,0,0],[0.6,1,0.6],[0,0,0]]
        # kernal = [0.6, 1, 0.6]
        kernal = [lamda1,lamda2,lamda1]
        kernal = torch.FloatTensor(kernal)
        # kernal = kernal.unsqueeze(0).unsqueeze(0)
        kernal = kernal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernal,requires_grad=False)

    def forward(self, x):
        # x = F.conv1d(x,self.weight,padding=1)
        x = F.conv1d(x, self.weight, padding=(0,1))
        return x


if __name__=='__main__':
    # usage
    mymodel = model_resnet50(num_clasess=50)
    mymodel.cuda()
    # prepare input
    input = Variable(torch.randn(4*16, 3,242, 242)).cuda()
    feature,output = mymodel(input)
    print(feature.shape,output.shape)
    # torch.Size([64, 2048]) torch.Size([64, 64])




