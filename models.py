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

        # self.fc3 = nn.Linear(3200, 512)
        # self.act3 = nn.ReLU()
        # self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(3200, n_species)
        self.act4 = nn.ReLU()
    
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

        # input 3200, output 512
        # x = self.act3(self.fc3(x))
        # x = self.drop3(x)
        
        # input 3200, output n_species
        x = self.fc4(x)
        x = self.act4(x)
        return x


from torch import nn

class cnn2(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(8, 4, kernel_size=3)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(14400, 5000)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(5000, n_species)
        self.act5 = nn.ReLU()
    
    def forward(self, x):
        # input n_featuresx128x128, output 32x126x126
        x = self.act1(self.conv1(x))
        x = self.drop1(x)

        # input 32x126x126, output 8x124x124
        x = self.act2(self.conv2(x))
        # input 8x124x124, output 8x62x62
        x = self.pool2(x)

        # input 8x62x62, output 4x60x60
        x = self.act3(self.conv3(x))
        x = self.drop3(x)

        # input 4x60x60, output 14400
        x = self.flat(x)

        # input 3200, output 512
        x = self.act4(self.fc4(x))
        x = self.drop4(x)
        
        # input 512, output n_species
        x = self.fc5(x)
        x = self.act5(x)
        return x
