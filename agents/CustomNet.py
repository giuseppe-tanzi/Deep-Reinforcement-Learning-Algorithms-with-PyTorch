import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        print("First conv: ", x.shape)
        x = self.relu2(self.conv2(x))
        print("Second conv: ", x.shape)
        x = x.view(32*8*8)  # Flatten the tensor
        print("Flattening: ", x.shape)
        x = self.relu3(self.fc1(x))
        print("First fc: ", x.shape)
        x = self.relu4(self.fc2(x))
        print("Second fc: ", x.shape)
        x = self.fc3(x)
        print("Third fc: ", x.shape)
        x = self.softmax(x)
        print("Softmax: ", x.shape)

        return x

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, last_activation=None):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 144)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(144, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        if last_activation is None:
            self.activation = nn.Softmax()
        else:
            self.activation = nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.activation(x)
        return x