import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(in_features=64*3*3, out_features=10)

    def forward(self, x):
        model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.pool3,
            nn.Flatten(),
            self.fc
        )
        return model(x)
        
class ConvNetVariable(nn.Module):
    def __init__(self, N: int, pool_type: str):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=N, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        if pool_type == 'max':
            self.pool4 = nn.AdaptiveMaxPool2d((1,1))
        elif pool_type == 'avg':
            self.pool4 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=N, out_features=10)

    def forward(self, x):
        model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.pool3,
            self.pool4,
            nn.Flatten(),
            self.fc
        )
        return model(x)