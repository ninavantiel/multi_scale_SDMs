from torch import nn
from math import floor

class MLP(nn.Module):

    def __init__(self, input_size, output_size, num_layers, width, dropout=0.0):
        super(MLP, self).__init__()

        self.output_size = output_size
        self.width = width

        layers = []

        layers.append(nn.Linear(input_size, width))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
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
            layers.append(nn.SiLU())
            layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))
            patch_size = floor((patch_size - kernel_size + 1) / 2)
            # layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(patch_size*patch_size*self.n_filters[-1], fc_width))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(fc_width, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x