from torch import nn
from torchvision import models
from math import floor

class MLP(nn.Module):

    def __init__(self, input_size, output_size, num_layers, width, dropout=0.0):
        super(MLP, self).__init__()

        self.output_size = output_size
        self.width = width

        layers = []

        layers.append(nn.Linear(input_size, width))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
    
        layers.append(nn.Linear(width, output_size))

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

    def __init__(self, input_feature_size, input_patch_size, output_size, 
                 num_conv_layers, n_filters, fc_width, kernel_size=3, pooling_size=1, dropout=0.0):
        super(ShallowCNN, self).__init__()

        self.n_filters = [input_feature_size] + n_filters
        self.output_size = output_size
        
        patch_size = input_patch_size
        layers = []

        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(self.n_filters[i], self.n_filters[i+1], kernel_size=kernel_size))
            layers.append(nn.BatchNorm2d(self.n_filters[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))
            patch_size = floor((patch_size - kernel_size + 1) / pooling_size)
            # layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(patch_size*patch_size*self.n_filters[-1], fc_width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(fc_width, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def get_resnet(n_input_channels, target_size, init_extra_channels=0):
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

    if n_input_channels == 4:
        weights = model.conv1.weight.data.clone()
        model.conv1 = nn.Conv2d(n_input_channels, 64, 
                                kernel_size=(7,7), stride=(2,2),
                                padding=(3,3), bias=False)
        # assume first three channels are RGB 
        model.conv1.weight.data[:, :3, :, :] = weights
        # for weights for 4th channel, use weights for channel "init_extra_channels"
        model.conv1.weight.data[:, -1, :, :] = weights[:, init_extra_channels, :, :]
        model.fc = nn.Linear(512, target_size)

        return model