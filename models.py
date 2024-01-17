from torch import nn

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
    
# class CNN(nn.Module):

#     def __init__(self, input_size, output_size, num_conv_layers, filter_size, num_fc_layers, width, dropout=0.0):
#         super(CNN, self).__init__()

#         self.output_size = output_size
#         self.width = width

#         layers = []

#         layers.append(nn.Conv2d(input_size))

#         layers.append(nn.Linear(input_size, width))
#         layers.append(nn.SiLU())

#         for _ in range(num_layers - 1):
#             layers.append(nn.BatchNorm1d(width))
#             layers.append(nn.Linear(width, width))
#             layers.append(nn.SiLU())
#             layers.append(nn.Dropout(p=dropout))
    
#         layers.append(nn.Linear(width, output_size))

#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x