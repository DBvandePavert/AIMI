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
from tqdm import tqdm
from data import get_loaders


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(11 * 31 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = x.float()
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 24 * 32 * 32),  # 11 * 31 * 32
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 32, 24))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = x.float()
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        x = x[:, :, :256, :192]  # Questionable move

        return x


def run(model_config, data_config):
    train_loader, val_loader = get_loaders(data_config)

    loss_fn = torch.nn.MSELoss()

    lr = 0.001

    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Initialize the networks
    latent_dim = 128

    encoder = Encoder(encoded_space_dim=latent_dim)
    decoder = Decoder(encoded_space_dim=latent_dim)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    # Training cycle
    num_epochs = 50
    history_da = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))

        # Training (use the training function)
        train_loss = train_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim)

        # Validation  (use the testing function)
        val_loss = test_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn)

        # Print Validation loss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss,
                                                                              val_loss))


# Training function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer):

    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []

    for data_batch in tqdm(iter(dataloader)):

        # Move tensor to the proper device
        image_batch = data_batch['source'].to(device)

        # Encode data
        encoded_data = encoder(image_batch)

        # Decode data
        decoded_data = decoder(encoded_data)

        # Evaluate loss
        loss = loss_fn(decoded_data, data_batch['target'])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print batch loss
        # print('\t partial train loss (single batch): %f' % loss.data)
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


# Testing function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn):

    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        val_loss = []
        for data_batch in tqdm(iter(dataloader)):

            # Move tensor to the proper device
            image_batch = data_batch['source'].to(device)

            # Encode data
            encoded_data = encoder(image_batch)

            # Decode data
            decoded_data = decoder(encoded_data)

            loss = loss_fn(decoded_data, data_batch['target'])
            val_loss.append(loss.detach().cpu().numpy())

    return np.mean(val_loss)


if __name__ == '__main__':
    model_params = sys.argv[1]
    data_params = sys.argv[2]

    model_config = yaml.safe_load(open(model_params))
    data_config = yaml.safe_load(open(data_params))

    run(model_config, data_config)
