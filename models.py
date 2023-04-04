from torch import nn

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(11552, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 100)
        self.act4 = nn.ReLU()
    
    def forward(self, x):
        # input 10x128x128, output 32x126x126
        x = self.act1(self.conv1(x))
        # input 32x126x126, output 32x42x42
        x = self.pool1(x)
        x = self.drop1(x)

        # input 32x42x42, output 32x38x38
        x = self.act2(self.conv2(x))
        # input 32x38x38, output 32x19x19
        x = self.pool2(x)
        # input 32x19x19, output 11552
        x = self.flat(x)
        # input 11552, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 100
        x = self.fc4(x)
        x = self.act4(x)
        return x
