import numpy as np

import torch
import torch.nn as nn

import sys
import yaml
import time
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from data import get_loaders

import matplotlib.pyplot as plt
import os
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU()
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(11 * 31 * 32, 128),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(128, 24 * 32 * 32),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 32, 24))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = x.float()
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)

        # x = torch.sigmoid(x)

        return x


def run(model_config, data_config):
    train_loader, val_loader = get_loaders(data_config)

    lr = 0.001

    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Initialize the networks
    latent_dim = 64

    encoder = Encoder(encoded_space_dim=latent_dim)
    decoder = Decoder(encoded_space_dim=latent_dim)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(f'Selected device: {device}')

    loss_fn = torch.nn.MSELoss()
    loss_fn = loss_fn.to(device)

    loss_fn_lpips = lpips.LPIPS(net='vgg')
    loss_fn_lpips = loss_fn_lpips.to(device)

    loss_fn_ssim = SSIM(data_range=255, size_average=True, channel=1)
    loss_fn_ssim = loss_fn_ssim.to(device)

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    # Training cycle
    num_epochs = 3
    history_da = {'train_loss': [], 'val_loss': [], 'val_loss_lpips': [], 'val_loss_ssim': []}

    # Pick out like 5 samples from the validation set

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

        #Validation  (use the testing function)
        val_loss = test_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch)
        
        # Validation  with LPIPS instead of MSE (use the testing function)
        val_loss_lpips = test_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn_lpips)

        val_loss_ssim = test_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=val_loader,
            loss_fn=loss_fn_ssim,
            ssim=True)

        # Once every 5 epochs or so get the output from those 5 picked samples
        # Save the output images to a folder
        # Test locally first to see if the output is in the right format/colorspace etc

        # Print Validation loss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        history_da['val_loss_lpips'].append(val_loss_lpips)
        history_da['val_loss_ssim'].append(val_loss_ssim)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f} \t val loss lpips {:.3f} \t val loss ssim {:.3f}'.format(epoch + 1, num_epochs, train_loss,
                                                                             val_loss, val_loss_lpips, val_loss_ssim))

    # Also save a graph of the loss over all epochs so we can see where it stop learning
    plt.plot(list(range(1, num_epochs+1)), history_da['train_loss'], label='Trainingloss')
    plt.savefig(final_directory + '/train_loss_plot.jpg')

    #Plotting the validation loss for the MSE and saving it to the directory
    plt.plot(list(range(1, num_epochs+1)), history_da['val_loss'], label='ValidationlossMSE')
    plt.savefig(final_directory + '/val_loss_MSE_plot.jpg')

    #Plotting the validation loss for the LPIPS and saving it to the directory
    plt.plot(list(range(1, num_epochs+1)), history_da['val_loss_lpips'], label='ValidationlossLPIPS')
    plt.savefig(final_directory + '/val_loss_lpips_plot.jpg')

    #Plotting the validation loss for the SSIM and saving it to the directory
    plt.plot(list(range(1, num_epochs+1)), history_da['val_loss_ssim'], label='ValidationlossSSIM')
    plt.savefig(final_directory + '/val_loss_SSIM_plot.jpg')

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

        # To device
        decoded_data = decoded_data.to(device)
        data_batch_loss = data_batch['target'].to(device)

        # Evaluate loss
        loss = loss_fn(decoded_data, data_batch_loss)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


# Testing function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn, epoch = None, ssim = False):

    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():

        val_loss = []
        for data_batch in tqdm(iter(dataloader)):

            # Move tensor to the proper device
            image_batch = data_batch['source'].to(device)

            # Encode data
            encoded_data = encoder(image_batch)

            # Decode data
            decoded_data = decoder(encoded_data)

            # To device
            decoded_data = decoded_data.to(device)
            data_batch_loss = data_batch['target'].to(device, dtype=torch.float32)
            if ssim:
                loss = 1 - loss_fn(decoded_data, data_batch_loss)
                loss = torch.mean(loss)
                val_loss.append(loss.detach().cpu().numpy())
            else:
                loss = loss_fn(decoded_data, data_batch_loss)
                loss = torch.mean(loss) # If using LPIPS the loss returns an array, so take the mean
                val_loss.append(loss.detach().cpu().numpy())

            if epoch and epoch % 10 == 0:
                plt.imsave(final_directory + f'/{epoch}-output.jpg', decoded_data[0].cpu().permute(1,2,0).detach().numpy()[:,:,0], cmap="gray")
                plt.imsave(final_directory + f'/{epoch}-target.jpg', data_batch_loss[0].cpu().permute(1,2,0).detach().numpy()[:,:,0], cmap="gray")


    return np.mean(val_loss)


if __name__ == '__main__':
    model_params = sys.argv[1]
    data_params = sys.argv[2]

    dir_name = time.strftime("%Y%m%d-%H%M%S")

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory + "/output/", dir_name).replace("\\", "/")

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    model_config = yaml.safe_load(open(model_params))
    data_config = yaml.safe_load(open(data_params))

    run(model_config, data_config)

