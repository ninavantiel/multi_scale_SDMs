from torch import nn


class cnn(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(3200, n_species)
    
    def forward(self, x):
        # input n_featuresx128x128, output 32x126x126
        x = self.act1(self.conv1(x))
        # input 32x126x126, output 32x42x42
        x = self.pool1(x)
        x = self.drop1(x)

        # input 32x42x42, output 8x40x40
        x = self.act2(self.conv2(x))
        # input 8x40x40, output 8x20x20
        x = self.pool2(x)
        # input 8x20x20, output 3200
        x = self.flat(x)
        
        # input 3200, output n_species
        x = self.fc4(x)
        return x

class cnn_batchnorm(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(3200, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, n_species)
    
    def forward(self, x):
        # input n_featuresx128x128, output 32x126x126
        x = self.act1(self.batchnorm1(self.conv1(x)))
        # input 32x126x126, output 32x42x42
        x = self.pool1(x)
        x = self.drop1(x)

        # input 32x42x42, output 8x40x40
        x = self.act2(self.batchnorm2(self.conv2(x)))
        # input 8x40x40, output 8x20x20
        x = self.pool2(x)
        # input 8x20x20, output 3200
        x = self.flat(x)

        # input 3200, output 512
        x = self.act3(self.batchnorm3(self.fc3(x)))
        x = self.drop3(x)
        
        # input 512, output n_species
        x = self.fc4(x)
        return x

class cnn_batchnorm_act(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(3200, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, n_species)
        self.act4 = nn.ReLU()
    
    def forward(self, x):
        # input n_featuresx128x128, output 32x126x126
        x = self.act1(self.batchnorm1(self.conv1(x)))
        # input 32x126x126, output 32x42x42
        x = self.pool1(x)
        x = self.drop1(x)

        # input 32x42x42, output 8x40x40
        x = self.act2(self.batchnorm2(self.conv2(x)))
        # input 8x40x40, output 8x20x20
        x = self.pool2(x)
        # input 8x20x20, output 3200
        x = self.flat(x)

        # input 3200, output 512
        x = self.act3(self.batchnorm3(self.fc3(x)))
        x = self.drop3(x)
        
        # input 512, output n_species
        x = self.act4(self.fc4(x))
        return x

class cnn_batchnorm_patchsize_20(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        # self.fc3 = nn.Linear(3200, 512)
        # self.batchnorm3 = nn.BatchNorm1d(512)
        # self.act3 = nn.ReLU()
        # self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, n_species)
    
    def forward(self, x):
        # input n_featuresx20x20, output 16x18x18
        x = self.act1(self.batchnorm1(self.conv1(x)))
        x = self.drop1(x)

        # input 16x18x18, output 8x16x16
        x = self.act2(self.batchnorm2(self.conv2(x)))
        # input 8x16x16, output 8x8x8
        x = self.pool2(x)
        # input 8x8x8, output 512
        x = self.flat(x)

        # # input 3200, output 512
        # x = self.act3(self.batchnorm3(self.fc3(x)))
        # x = self.drop3(x)
        
        # input 512, output n_species
        x = self.fc4(x)
        return x
    
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
