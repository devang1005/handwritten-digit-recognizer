import torch
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(128*3*3,256),
            nn.ReLU(),
            nn.Linear(256,10)

        )

    def forward(self,x):

        x = self.conv1(x)
        x = self.fc(x)

        return x