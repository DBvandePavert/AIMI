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
import sys
import yaml
from utils import *
from data import get_loaders
from network import skip
torch.nn.Module.add = add_module

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def run(data_config):
    train_loader, _ = get_loaders(data_config)
    runs = []
    lr = 0.001
    input_depth = 32
    img_shape = 1
    num_epochs = 5
    show_every = 2

    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Check if the GPU is available
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for data_batch in iter(train_loader):
        # Define result dict
        results = {
            "train_loss": [],
            "val_loss": None,
            "outputs_gif": [],
            "output": None
        }

        # Get images
        source = data_batch["source"][0]
        target = data_batch["target"][0]


        # Create masked image
        masked = np.zeros((256,192), dtype = float)
        masked[::,1::2] = source / 255
        masked = np.expand_dims(masked, axis=0)

        # Create mask
        mask = np.zeros((256,192), dtype = float)
        ones = np.ones((256,96), dtype = float)
        mask[::,1::2] = ones
        mask = np.expand_dims(mask, axis=0)

        # Prepare target
        target = target / 255

        # Create tensors
        masked_tensor = torch.tensor(masked).type(dtype)
        mask_tensor = torch.tensor(mask).type(dtype)
        target_tensor = target

        # Define loss functions
        loss_fn_train = torch.nn.MSELoss()
        loss_fn_test = torch.nn.MSELoss()


        # Initialize the networks
        net = skip(input_depth, img_shape, 
            num_channels_down = [128] * 5,
            num_channels_up =   [128] * 5,
            num_channels_skip =    [128] * 5,  
            filter_size_up = 3, filter_size_down = 3, 
            upsample_mode='bilinear', filter_skip_size=1,
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU'
        ).type(dtype)

        # Define optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-05)

        # Create initial input
        net_input = (0.1) * torch.rand((1,32,256,192))

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
                results['outputs_gif'] = out.cpu().permute(1,2,0).detach().numpy()
                # plt.imsave('testImg', out.cpu().permute(1,2,0).detach().numpy()[:,:,0] * 255, cmap="gray")

            # Set weights
            train_loss.backward()

            # Regularization
            optimizer.step()

            net_input = net_input + (1 / (30)) * torch.randn_like(net_input)
        
        # Validation loss
        val_loss = loss_fn_test(out, target_tensor)
        results['val_loss'] = val_loss.item()
        results['output'] = out.cpu().permute(1,2,0).detach().numpy()

        runs.append(results)
        break

    return runs

if __name__ == '__main__':
    data_params = sys.argv[1]

    data_config = yaml.safe_load(open(data_params))

    runs = run(data_config)
    print(runs)
