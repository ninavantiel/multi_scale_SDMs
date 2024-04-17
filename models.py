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
    def __init__(self, in_channels, out_channels, kernel_sizes, dilations, pooling_sizes, n_linear_layers, target_size):
        super(ASPP_branch, self).__init__()

        conv_layers = []
        receptive_field = 1
        for i, (k, d, p) in enumerate(zip(kernel_sizes, dilations, pooling_sizes)):
            if i == 0: 
                conv_layers.append(nn.Conv2d(in_channels, out_channels, k, dilation=d)) #, padding = (k-1)//2)
            else:
                conv_layers.append(nn.Conv2d(out_channels, out_channels, k, dilation=d)) #padding = (k-1)//2),
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=p, stride=p))
            receptive_field = (receptive_field + (k-1)*d) * p
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
        self.out_n_channels = n_filters[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MultiResolutionModel(nn.Module):
    def __init__(self, in_channels, in_patch_size, target_size, cnn_params, aspp_params):
        super(MultiResolutionModel, self).__init__()        
        self.target_size = target_size

        self.backbone = CNN(
            in_channels, in_patch_size, cnn_params['n_filters'], cnn_params['kernel_sizes'], 
            cnn_params['paddings'], cnn_params['pooling_sizes']) 

        self.aspp_branches = nn.ModuleList([ASPP_branch(
            self.backbone.out_n_channels, aspp_params['out_channels'], 
            k, d, p, aspp_params['n_linear_layers'], aspp_params['out_size']
        ) for k, d, p in zip(aspp_params['kernel_sizes'], aspp_params['dilations'], aspp_params['pooling_sizes'])])

        self.linear = nn.Linear(aspp_params['out_size']*len(aspp_params['kernel_sizes']), target_size)
        
    def forward(self, x):
        x = self.backbone(x)
        xlist = [aspp(x) for aspp in self.aspp_branches]
        x = torch.cat(xlist, dim=1)
        x = self.linear(x)
        return x
    
class MultiResolutionAutoencoder(MultiResolutionModel):
    def __init__(self, in_channels, in_patch_size, target_size, cnn_params, aspp_params):
        super(MultiResolutionAutoencoder, self).__init__(in_channels, in_patch_size, target_size, cnn_params, aspp_params)
        
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

def make_model(model_dict):
    assert {'input_shape', 'output_shape'}.issubset(set(model_dict.keys()))

    if model_dict['model_name'] == 'MLP':
        param_names = {'n_layers', 'width', 'dropout'}
        assert param_names.issubset(set(model_dict.keys()))

        model = MLP(model_dict['input_shape'][0],
                    model_dict['output_shape'], 
                    model_dict['n_layers'], 
                    model_dict['width'], 
                    model_dict['dropout'])
        
    elif model_dict['model_name'] == 'CNN':
        param_names = {
            'patch_size', 'n_conv_layers', 'n_filters', 'width', 'kernel_size', 
            'padding', 'pooling_size', 'dropout', 'pool_only_last'
        }
        assert param_names.issubset(set(model_dict.keys()))
        assert model_dict['n_conv_layers'] == len(model_dict['n_filters'])

        model = ShallowCNN(model_dict['input_shape'][0],
                           model_dict['patch_size'], 
                           model_dict['output_shape'],
                           model_dict['n_conv_layers'], 
                           model_dict['n_filters'], 
                           model_dict['width'], 
                           model_dict['kernel_size'], 
                           model_dict['padding'],
                           model_dict['pooling_size'], 
                           model_dict['dropout'],
                           model_dict['pool_only_last'])
        
    elif model_dict['model_name'] == 'ResNet':
        assert 'pretrained' in list(model_dict.keys())

        model = get_resnet(
            model_dict['output_shape'], 
            model_dict['input_shape'][0], 
            model_dict['pretrained'])
        
    elif model_dict['model_name'] in ['MultiResolutionModel', 'MultiResolutionAutoencoder']:
        param_names = {'patch_size', 'backbone_params', 'aspp_params'}
        assert param_names.issubset(set(model_dict.keys()))

        backbone_param_names = {'n_filters', 'kernel_sizes', 'paddings', 'pooling_sizes'}
        assert backbone_param_names.issubset(set(model_dict['backbone_params'].keys()))
        aspp_param_names = {'out_channels', 'out_size', 'kernel_sizes', 'dilations', 'pooling_sizes', 'n_linear_layers'}
        assert aspp_param_names.issubset(set(model_dict['aspp_params'].keys()))

        if model_dict['model_name'] == 'MultiResolutionModel':
            model = MultiResolutionModel(
                model_dict['input_shape'][0],
                model_dict['patch_size'],
                model_dict['output_shape'],
                model_dict['backbone_params'], 
                model_dict['aspp_params'])
            
        elif model_dict['model_name'] == 'MultiResolutionAutoencoder':
                model = MultiResolutionAutoencoder(
                model_dict['input_shape'][0],
                model_dict['patch_size'],
                model_dict['output_shape'],
                model_dict['backbone_params'], 
                model_dict['aspp_params'])

    return model