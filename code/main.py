"""
Main file for UvA course Artificial intelligence for Medical imaging

- Daniël van de Pavert
-
-
-
-
"""

import torch
import pytorch_lightning as pl
from data import get_loaders


def run(config):

    # Set seed

    # Get dataloaders
    train_loader, val_loader = get_loaders(config)

    # Get model

    # Get logger

    # Get Trainer

    # Train


def get_config():

    config = {
        "path_train": "data/data_train_plus_test_sourceres/train/",
        "path_test": "data/data_train_plus_test_sourceres/",
        "N": 3,
        'verbose': 1,
        "batchsize": 1
    }

    return config


if __name__ == '__main__':

    config = get_config()
    run(config)
