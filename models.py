import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
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
                 kernel_size=3, padding=0, pooling_size=2, dropout=0.5, pool_only_last=False):
        super(ShallowCNN, self).__init__()

        self.n_filters = [input_feature_size] + n_filters
        self.target_size = target_size
        
        patch_size = input_patch_size
        layers = []

        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(self.n_filters[i], self.n_filters[i+1], kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(self.n_filters[i+1]))
            layers.append(nn.ReLU())
            if not pool_only_last:  
                layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))
                patch_size = floor((patch_size - kernel_size + 2*padding + 1) / pooling_size)
            elif i == num_conv_layers-1:
                layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))
                patch_size = floor((patch_size - kernel_size + 2*padding + 1) / pooling_size)
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
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

class ASPP_branch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, pooling_sizes, n_linear_layers, target_size):
        super(ASPP_branch, self).__init__()

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
    
class decoder_branch(nn.Module):
    def __init__(self, n_linear_layers, in_size, n_channels, out_channels, kernel_sizes, pooling_sizes):
        super(decoder_branch, self).__init__()

        linear_layers = []
        linear_layers.append(nn.Linear(in_size, n_channels))
        linear_layers.append(nn.ReLU())
        for _ in range(n_linear_layers-1):
            linear_layers.append(nn.Linear(n_channels, n_channels))
            linear_layers.append(nn.ReLU())
        self.linear_layers = nn.Sequential(*linear_layers)

        deconv_layers = []
        for k, p in zip(kernel_sizes[:-1], pooling_sizes[:-1]):
            deconv_layers.append(nn.Upsample(scale_factor=p, mode='bilinear', align_corners=True))
            deconv_layers.append(nn.ConvTranspose2d(n_channels, n_channels, k))
            deconv_layers.append(nn.ReLU())
        deconv_layers.append(nn.Upsample(scale_factor=pooling_sizes[-1], mode='bilinear', align_corners=True))
        deconv_layers.append(nn.ConvTranspose2d(n_channels, out_channels, kernel_sizes[-1]))
        deconv_layers.append(nn.ReLU())
        self.deconv_layers = nn.Sequential(*deconv_layers)

    def forward(self, x):
        x = self.linear_layers(x)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.deconv_layers(x)
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
    
class ResNet_3layers(nn.Module):
    def __init__(self, in_channels, pretrained): 
        super(ResNet_3layers, self).__init__()
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
    def __init__(self, in_channels, in_patch_size, target_size, backbone, backbone_params, aspp_params):
        super(MultiResolutionModel, self).__init__()        
        self.target_size = target_size

        if backbone == 'CNN':
            self.backbone = CNN(
                in_channels, in_patch_size, backbone_params['n_filters'], backbone_params['kernel_sizes'], 
                backbone_params['paddings'], backbone_params['pooling_sizes']) 
        elif backbone == 'ResNet':
            self.backbone = ResNet_3layers(in_channels, backbone_params['pretrained'])

        self.aspp_branches = nn.ModuleList([ASPP_branch(
            self.backbone.out_channels, aspp_params['out_channels'], 
            k, d, p, aspp_params['n_linear_layers'], aspp_params['out_size']
        ) for k, d, p in zip(aspp_params['kernel_sizes'], aspp_params['strides'], aspp_params['pooling_sizes'])])

        self.linear = nn.Linear(aspp_params['out_size']*len(aspp_params['kernel_sizes']), target_size)
        # self.linear = nn.Linear(len(aspp_params['kernel_sizes']), 1)
        
    def forward(self, x):
        x = self.backbone(x)
        xlist = [aspp(x) for aspp in self.aspp_branches]
        x = torch.cat(xlist, dim=1)
        # x = torch.stack(xlist, dim=2)
        x = torch.squeeze(self.linear(x))
        return x
    
class MultiResolutionAutoencoder(MultiResolutionModel):
    def __init__(self, in_channels, in_patch_size, target_size, backbone, backbone_params, aspp_params):
        super(MultiResolutionAutoencoder, self).__init__(backbone, in_channels, in_patch_size, target_size, backbone, backbone_params, aspp_params)
        
        self.decoder_branches = nn.ModuleList([decoder_branch(
            aspp_params['n_linear_layers'], self.target_size, 
            aspp_params['out_channels'], in_channels, k, p
        ) for k, p in zip(aspp_params['kernel_sizes'][::-1], aspp_params['pooling_sizes'][::-1])])
   
    def forward(self, x):
        x = self.backbone(x)
        xlist = [aspp(x) for aspp in self.aspp_branches]
        x = torch.cat(xlist, dim=1)
        x = self.linear(x)

        xlist = [decoder(xl) for xl, decoder in zip(xlist, self.decoder_branches)]
        return x, xlist
