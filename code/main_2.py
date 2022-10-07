import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# %matplotlib inline
import torch.nn as nn
import torch.nn.functional as F
from data import get_loaders
import sys
import yaml
import tqdm


# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 4, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(8, 1, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.sigmoid(self.t_conv5(x))

        return x

def run(model_config, data_config):
    # Instantiate the model
    model = ConvAutoencoder()
    print(model)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def get_device():
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    device = get_device()
    print(device)
    model.to(device)

    train_loader, val_loader = get_loaders(data_config)

    train(train_loader, device, optimizer, model, criterion)

def train(train_loader, device, optimizer, model, criterion):
    # Epochs
    n_epochs = 10

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        # Training
        for data in train_loader:
            images = data['source']
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


if __name__ == '__main__':

    model_params = sys.argv[1]
    data_params = sys.argv[2]

    model_config = yaml.safe_load(open(model_params))
    data_config = yaml.safe_load(open(data_params))

    run(model_config, data_config)