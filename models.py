import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
from math import floor
import numpy as np

class MultimodalModel(nn.Module):
    def __init__(self, modelA, modelB, target_size):
        super(MultimodalModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc = nn.Linear(modelA.out_size + modelB.out_size, target_size)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

class spatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, pooling_sizes, n_linear_layers, target_size):
        super(spatialBlock, self).__init__()

        conv_layers = []
        receptive_field = 1
        for i, (k, s, p) in enumerate(zip(kernel_sizes, strides, pooling_sizes)):
            if i == 0: 
                conv_layers.append(nn.Conv2d(in_channels, out_channels, k, stride=s)) #padding = (k-1)//2)
            else:
                conv_layers.append(nn.Conv2d(out_channels, out_channels, k, stride=s)) #padding = (k-1)//2)
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=p, stride=p))
            receptive_field = ((receptive_field*s)+k-s) * p
        self.conv_layers = nn.Sequential(*conv_layers)
        self.receptive_field = receptive_field

        self.crop = transforms.CenterCrop(1)
        
        linear_layers = []
        for _ in range(n_linear_layers-1):
            linear_layers.append(nn.Linear(out_channels, out_channels))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(out_channels, target_size))
        linear_layers.append(nn.ReLU())
        self.linear_layers = nn.Sequential(*linear_layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.crop(x).squeeze()
        x = self.linear_layers(x)
        return x
  

class CNN(nn.Module):
    def __init__(
            self, in_channels, in_patch_size, n_filters, kernel_sizes, paddings, pooling_sizes): 
        super(CNN, self).__init__()
        patch_size = in_patch_size

        layers = []
        for f_in, f, k, p, pool in zip([in_channels]+n_filters[:-1], n_filters, kernel_sizes, paddings, pooling_sizes):
            layers.append(nn.Conv2d(f_in, f, kernel_size=k, padding=p))
            layers.append(nn.BatchNorm2d(f))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
            patch_size = floor((patch_size + 2*p - k + 1) / pool)

        self.layers = nn.Sequential(*layers)
        self.out_patch_size = patch_size
        self.out_channels = n_filters[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ResNet_9layers(nn.Module):
    def __init__(self, in_channels, pretrained): 
        super(ResNet_9layers, self).__init__()
        if pretrained:
            model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            weights = model.conv1.weight.data.clone()
        else:
            model = models.resnet18()

        # if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            if pretrained:
                self.layer0[0].weight.data[:, :3, :, :] = weights
                self.layer0[0].weight.data[:, 3, :, :] = weights[:, 0, :, :]
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        # self.layer3 = model.layer3
        # self.layer4 = model.layer4
        # self.fc = nn.Linear(512, target_size)

        self.out_channels = self.layer2[-1].conv2.out_channels

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class MultiResolutionModel(nn.Module):
    def __init__(self, in_channels, in_patch_size, target_size, backbone, backbone_params, spatial_block_params):
        super(MultiResolutionModel, self).__init__()
        if target_size is None:
            self.linear_layer = False
            self.out_size = spatial_block_params['out_size']*len(spatial_block_params['kernel_sizes'])
        else:
            self.linear_layer = True
            self.out_size = target_size

        if backbone == 'CNN':
            self.backbone = CNN(
                in_channels, in_patch_size, backbone_params['n_filters'], backbone_params['kernel_sizes'], 
                backbone_params['paddings'], backbone_params['pooling_sizes']) 
        elif backbone == 'ResNet':
            self.backbone = ResNet_9layers(in_channels, backbone_params['pretrained'])

        self.spatial_branches = nn.ModuleList([spatialBlock(
            self.backbone.out_channels, spatial_block_params['out_channels'], 
            k, d, p, spatial_block_params['n_linear_layers'], spatial_block_params['out_size']
        ) for k, d, p in zip(spatial_block_params['kernel_sizes'], spatial_block_params['strides'], spatial_block_params['pooling_sizes'])])
    
        if self.linear_layer:
            self.linear = nn.Linear(spatial_block_params['out_size']*len(spatial_block_params['kernel_sizes']), target_size)
        
    def forward(self, x):
        x = self.backbone(x)
        xlist = [branch(x) for branch in self.spatial_branches]
        x = torch.cat(xlist, dim=1)
        # x = torch.stack(xlist, dim=2)
        if self.linear_layer:
            x = torch.squeeze(self.linear(x))
        return x
    