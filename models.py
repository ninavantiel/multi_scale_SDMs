import torch
from torch import nn
from torchvision import models
from math import floor
import numpy as np

class MLP(nn.Module):

    def __init__(self, input_size, target_size, num_layers, width, dropout=0.0):
        super(MLP, self).__init__()

        self.target_size = target_size
        self.width = width

        layers = []

        layers.append(nn.Linear(input_size, width))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
    
        layers.append(nn.Linear(width, target_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ShallowCNN(nn.Module):
    '''
    Inspired by the shallow convolutional neural network (SCNNB) from:
    Lei et al. 2020. Shallow convolutional neural network for image classification.
    https://doi.org/10.1007/s42452-019-1903-4
    '''

    def __init__(self, input_feature_size, input_patch_size, target_size, 
                 num_conv_layers=2, n_filters=[32,64], fc_width=1280, 
                 kernel_size=3, pooling_size=2, dropout=0.5, pool_only_last=False):
        super(ShallowCNN, self).__init__()

        self.n_filters = [input_feature_size] + n_filters
        self.target_size = target_size
        
        patch_size = input_patch_size
        layers = []

        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(self.n_filters[i], self.n_filters[i+1], kernel_size=kernel_size))
            layers.append(nn.BatchNorm2d(self.n_filters[i+1]))
            layers.append(nn.ReLU())
            if not pool_only_last:  
                layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))
                patch_size = floor((patch_size - kernel_size + 1) / pooling_size)
            elif i == num_conv_layers-1:
                layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))
                patch_size = floor((patch_size - kernel_size + 1) / pooling_size)
            else:
                patch_size = patch_size - kernel_size + 1
                
        layers.append(nn.Flatten())
        layers.append(nn.Linear(patch_size*patch_size*self.n_filters[-1], fc_width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(fc_width, target_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def get_resnet(target_size, n_input_channels=4, pretrained=True, init_extra_channels=0):
    if pretrained:
        assert n_input_channels == 3 or n_input_channels == 4
        # get restnet18 model with pretrained weights wtih 3 input channels
        model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        # add one input channel for 4th band
        if n_input_channels == 4:
            weights = model.conv1.weight.data.clone()
            model.conv1 = nn.Conv2d(n_input_channels, 64, 
                                    kernel_size=(7,7), stride=(2,2),
                                    padding=(3,3), bias=False)
            # assume first three channels are RGB 
            model.conv1.weight.data[:, :3, :, :] = weights
            # for weights for 4th channel, use weights for channel "init_extra_channels" (0=R, 1=G, 2=B)
            model.conv1.weight.data[:, -1, :, :] = weights[:, init_extra_channels, :, :]

    else:
        # get resnet18 model without pretrained weights
        model = models.resnet18()
        # adapt number of input channels in first convolutional layer
        model.conv1 = nn.Conv2d(n_input_channels, 64,
                                kernel_size=(7,7), stride=(2,2),
                                padding=(3,3), bias=False)
    
    # adapt output size of the last layer
    model.fc = nn.Linear(512, target_size)

    return model

class MultimodalModel(nn.Module):
    def __init__(self, modelA, modelB, target_size, outsizeA, outsizeB):
        super(MultimodalModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc = nn.Linear(outsizeA + outsizeB, target_size)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        #?? softmax before concat??
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

def aspp_branch(in_channels, out_channels, kernel_size, dilation):
    '''
    As implemented in:
    https://github.com/yassouali/pytorch-segmentation/blob/master/models/deeplabv3_plus.py
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

class ASPP(nn.Module):
    '''
    Adapted from:
    https://github.com/yassouali/pytorch-segmentation/blob/master/models/deeplabv3_plus.py
    '''
    def __init__(self, in_channels, in_patch_size, out_channels, kernel_sizes, dilations, dropout, device):
        super(ASPP, self).__init__()
        self.subpatch_sizes = [(d-1)*(k-1) + k for k, d in zip(kernel_sizes, dilations)]
        self.center_idx = in_patch_size // 2
        self.imins = [int(self.center_idx - (s-1)/2) for s in self.subpatch_sizes]
        self.imaxs = [int(self.center_idx + (s-1)/2 + 1) for s in self.subpatch_sizes]
        assert (np.array(self.imins) >= 0).all()
        assert (np.array(self.imaxs) < in_patch_size).all()

        self.aspp_branches = [aspp_branch(in_channels, out_channels, k, d).to(device) for k, d in zip(kernel_sizes, dilations)]

    def forward(self, x):
        x_in = [x[:, :, imin:imax, imin:imax] for imin, imax in zip(self.imins, self.imaxs)]
        x = [aspp(xi) for aspp, xi in zip(self.aspp_branches, x_in)]
        x = torch.cat(x, dim=1).squeeze()
        return x
    
class CNN(nn.Module):
    def __init__(
            self, in_channels, in_patch_size, num_conv_layers, n_filters, 
            kernel_size, padding, pooling_size):
        super(CNN, self).__init__()
        self.n_filters = [in_channels] + n_filters
        patch_size = in_patch_size
        layers = []
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(self.n_filters[i], self.n_filters[i+1], kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(self.n_filters[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))
            patch_size = floor((patch_size + 2*padding - kernel_size + 1) / pooling_size)

        self.layers = nn.Sequential(*layers)
        self.out_patch_size = patch_size
        self.out_n_channels = self.n_filters[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MultiResolutionModel(nn.Module):
    def __init__(self, in_channels, in_patch_size, target_size, backbone, backbone_params,
                 aspp_out_channels, aspp_kernel_sizes, aspp_dilations, dropout, device):
        super(MultiResolutionModel, self).__init__()        
        self.target_size = target_size

        if backbone == 'CNN':
            self.backbone = CNN(in_channels, in_patch_size, 
                                backbone_params['n_conv_layers'], 
                                backbone_params['n_filters'], 
                                backbone_params['kernel_size'], 
                                backbone_params['padding'], 
                                backbone_params['pooling_size'])
        else:
            print('backbone not supported')
            return 
        
        self.aspp_block = ASPP(
            self.backbone.out_n_channels,
            self.backbone.out_patch_size, 
            aspp_out_channels, 
            aspp_kernel_sizes, 
            aspp_dilations, 
            dropout, 
            device)
        self.linear = nn.Linear(aspp_out_channels*len(aspp_dilations), target_size)
         
    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp_block(x)
        x = self.linear(x)
        return x