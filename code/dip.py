import os
import time
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
import torch.nn as nn
import torch.nn.functional as F
import sys
import yaml
from utils import *
from data import get_loaders
from network import skip
import lpips
from pytorch_msssim import SSIM
torch.nn.Module.add = add_module

def run(data_config):
    train_loader, _ = get_loaders(data_config)
    runs = []
    lr = 0.001
    input_depth = 32
    img_shape = 1
    num_epochs = 6000 if torch.cuda.is_available() else 2
    show_every = 100 if torch.cuda.is_available() else 1

    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_fn_lpips = lpips.LPIPS(net='vgg')

    for data_batch in iter(train_loader):
        # data_batch = train_loader.dataset.__getitem__(180) # Comment for complete run
        
        
        # Define result dict
        results = {
            'train_loss': [],
            'outputs_gif': [],
            'val_loss': None,
            'val_loss_lpips': [],
            'val_loss_ssim': [],
            'val_loss_mae': [],
            'output': None,
        }

        # Get images
        source = data_batch["source"][0]
        target = data_batch["target"][0]

        # plt.imsave(final_directory + '/source.jpg', source, cmap="gray")

        source = np.array(source.squeeze(0)) # Comment out for complete run
        target = np.array(target.squeeze(0)) # Comment out for complete run

        # Create masked image
        masked = np.zeros((256,192), dtype = float)
        masked[::,1::2] = source
        masked = np.expand_dims(masked, axis=0)

        # Create mask
        mask = np.zeros((256,192), dtype = float)
        ones = np.ones((256,96), dtype = float)
        mask[::,1::2] = ones
        mask = np.expand_dims(mask, axis=0)

        # Prepare target
        target = np.expand_dims(target, axis=0)

        # Create tensors
        masked_tensor = torch.tensor(masked)
        mask_tensor = torch.tensor(mask)
        target_tensor = torch.tensor(target)

        # Move to device
        masked_tensor = masked_tensor.to(device, dtype=torch.float32)
        mask_tensor = mask_tensor.to(device, dtype=torch.float32)
        target_tensor = target_tensor.to(device, dtype=torch.float32)

        # Sanity
        plt.imsave(final_directory + '/masked.jpg', masked_tensor.cpu().permute(1,2,0).detach().numpy()[:,:,0], cmap="gray")
        plt.imsave(final_directory + '/mask.jpg', mask_tensor.cpu().permute(1,2,0).detach().numpy()[:,:,0], cmap="gray")
        plt.imsave(final_directory + '/target.jpg', target_tensor.cpu().permute(1,2,0).detach().numpy()[:,:,0], cmap="gray")

        # Define loss functions
        loss_fn_train = torch.nn.MSELoss().to(device)
        loss_fn_test = torch.nn.MSELoss().to(device)
        loss_fn_lpips = loss_fn_lpips.to(device)
        loss_fn_ssim = SSIM(data_range=255, size_average=True, channel=1)
        loss_fn_ssim = loss_fn_ssim.to(device)
        loss_fn_mae = torch.nn.L1Loss()
        loss_fn_mae = loss_fn_mae.to(device)

        # Initialize the networks
        net = skip(input_depth, img_shape, 
            num_channels_down = [128] * 5,
            num_channels_up =   [128] * 5,
            num_channels_skip =    [128] * 5,  
            filter_size_up = 3, filter_size_down = 3, 
            upsample_mode='bilinear', filter_skip_size=1,
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU'
        ).to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-05)

        # Create initial input
        net_input = ((0.1) * torch.rand((1,32,256,192))).to(device)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            out = net(net_input)
            out = out.squeeze(0)

            # Training loss
            train_loss = loss_fn_train(out * mask_tensor, masked_tensor)
            results['train_loss'].append(train_loss.item())

            if epoch % show_every == 0:
                print('EPOCH %d/%d' % (epoch + 1, num_epochs))
                print("Loss", train_loss.item())
                # results['outputs_gif'] = out.cpu().permute(1,2,0).detach().numpy()
                # plt.imsave('testImg', out.cpu().permute(1,2,0).detach().numpy()[:,:,0] * 255, cmap="gray")
                # plt.imsave(final_directory + "/gifs" + f'/{epoch}.jpg', out.cpu().permute(1,2,0).detach().numpy()[:,:,0], cmap="gray")

            # Set weights
            train_loss.backward()

            # Regularization
            optimizer.step()

            net_input = net_input + (1 / (30)) * torch.randn_like(net_input)
        
        # Validation loss
        val_loss = loss_fn_test(out, target_tensor)      
        results['val_loss'] = val_loss.item()

        val_loss_lpips = loss_fn_lpips(out, target_tensor)
        results['val_loss_lpips'] = val_loss_lpips.item()

        if len(out.shape) == 3:
            ssim_out = out.unsqueeze(0)
        else:
            ssim_out = out
        if len(target_tensor.shape) == 3:
            ssim_target_tensor = target_tensor.unsqueeze(0)
        else:
            ssim_target_tensor = target_tensor     
        val_loss_ssim = loss_fn_ssim(ssim_out, ssim_target_tensor)
        results['val_loss_ssim'] = val_loss_ssim.item()

        val_loss_mae = loss_fn_mae(out, target_tensor)
        results['val_loss_mae'] = val_loss_mae.item()

        results['output'] = out.cpu().permute(1,2,0).detach().numpy() / 255

        runs.append(results)
        # break # Comment for complete run

    return runs

if __name__ == '__main__':
    data_params = sys.argv[1]

    dir_name = time.strftime("%Y%m%d-%H%M%S")

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory + "/output/", dir_name).replace("\\", "/")

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        os.makedirs(final_directory + "/gifs")

    data_config = yaml.safe_load(open(data_params))

    runs = run(data_config)

    validation_loss = []
    validation_loss_lpips = []
    validation_loss_ssim = []
    validation_loss_mae = []
    for run in runs:
        validation_loss.append(run['val_loss'])
        validation_loss_lpips.append(run['val_loss_lpips'])
        validation_loss_ssim.append(run['val_loss_ssim'])
        validation_loss_mae.append(run['val_loss_mae'])
    print("Validation loss: ", np.mean(validation_loss))
    print("Validation lpips loss: ", np.mean(validation_loss_lpips))
    print("Validation ssim loss: ", np.mean(validation_loss_ssim))
    print("Validation mae loss: ", np.mean(validation_loss_mae))
    
    # plt.imsave(final_directory + '/final.jpg', (runs[0]['output'])[:, :, 0], cmap="gray")


