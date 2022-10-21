""""
File to hold the data related functions

"""

import string
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import SimpleITK as sitk
import os
from typing import List, Tuple

SET_A_MAX = 0
SET_B_MAX = 0
SET_C_MAX = 0
class SR_Dataset(Dataset):
    """
    Dataset object returning data pairs

    Arguments:
        config (dict):  configuration dict holding hyperparameters
        data (List) :  List of data pairs
    """
    def __init__(self, config, data):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_loaders(config):
    # Messy combination function, should rewrite

    data = read_files(config) 
    data = remove_empty_scans(config, data)
    padded = add_padding(config, data)
    train_pairs, val_pairs = create_pairs(config, padded)

    train_set = SR_Dataset(config, train_pairs)
    val_set = SR_Dataset(config, val_pairs)

    train_loader = DataLoader(train_set, batch_size=config['batchsize'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['batchsize'], shuffle=True, num_workers=0)

    return train_loader, val_loader

def calculate_max_pixel_values(f: string):
    """
    Function to calculate the max pixel values of the sets in the data

    Arguments:
        f (string): filename of image file

    """
    global SET_A_MAX, SET_B_MAX, SET_C_MAX

    source = sitk.GetArrayFromImage(sitk.ReadImage('source_resolution/' + f))

    if 'SET_A' in f:
        if source.max() > SET_A_MAX:
            SET_A_MAX = source.max()
    if 'SET_B' in f:
        if source.max() > SET_B_MAX:
            SET_B_MAX = source.max()
    if 'SET_C' in f:
        if source.max() > SET_C_MAX:
            SET_C_MAX = source.max()

def normalize_values(f: string):
    """
    Function to normalize the data values between 0 and 1

    Arguments:
        f (string): filename of image file

    """
    global SET_A_MAX, SET_B_MAX, SET_C_MAX

    source = sitk.GetArrayFromImage(sitk.ReadImage('source_resolution/' + f))
    target = sitk.GetArrayFromImage(sitk.ReadImage('target_resolution/' + f))

    if 'SET_A' in f:
        source = source / SET_A_MAX
        target = target / SET_A_MAX
    if 'SET_B' in f:
        source = source / SET_B_MAX
        target = target / SET_B_MAX
    if 'SET_C' in f:
        source = source / SET_C_MAX
        target = target / SET_C_MAX

    return source, target


def read_files(config: dict) -> List[List]:
    """
    Function to read the files from the disk and organize them by type

    Arguments:
        config (dict): configuration dict holding hyperparameters

    """
    if config['verbose']:
        print('Loading data')

    # Get train files
    os.chdir(config['path_train'])
    os.chdir('source_resolution')
    files = os.listdir()
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')

    # Create train set
    train_set = [f for f in files]

    # Get test files
    os.chdir(config['path_test'])
    os.chdir('source_resolution')
    files = os.listdir()
    os.chdir('..')


    # Create test set
    test_set = [f for f in files]

    # Create data sets
    data = [[], []]
    sets = [train_set, test_set]

    # Calculate max pixel values per set (per MRI machine type)
    for i in range(len(data)):
        # Go back to correct directory
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')

        # Set correct working directory
        if i == 0:
            os.chdir(config['path_train'])
        else:
            os.chdir(config['path_test'])

        for f in sets[i]:
            calculate_max_pixel_values(f)

    if config['verbose']:
        print(f"Calculated max values for normalisation: {SET_A_MAX, SET_B_MAX, SET_C_MAX}")
   
    # Normalize values of scans according to max pizxel values
    for i in range(len(data)):
        # Go back to correct directory
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
    
        # Set correct working directory
        if i == 0:
            os.chdir(config['path_train'])
        else:
            os.chdir(config['path_test'])
        
        # Normalize and add scans to datasets
        for f in sets[i]:
            source, target = normalize_values(f)

            patient = {
                "source": source,
                "target": target
            }

            data[i].append(patient)


    if config['verbose']:
        print(
            f'''
        Found data:
            - set train: {len(data[0])}
            - set test: {len(data[1])}
        '''
        )

    return data


def add_padding(config: dict, data: List) -> List[List]:
    """
    Adds padding to the data in order to comply with the N neighbours taken later

    Arguments:
        config (dict):  configuration dict holding hyperparameters
        data (List) :  List holding data from the various sets
    """
    if config['verbose']:
        print('Padding data')

    if int(config['N']) == 1:
        return data

    padding = int((int(config['N']) - 1) / 2)
    padded = [[], []]

    for i in range(len(data)):
        for patient in data[i]:
            patient['source'] = np.pad(patient['source'], ((padding, padding), (0, 0), (0, 0)), 'constant')
            patient['target'] = np.pad(patient['target'], ((padding, padding), (0, 0), (0, 0)), 'constant')
            padded[i].append(patient)

    return padded


def create_pairs(config: dict, data: List) -> (List[dict], List[dict]):
    """
    Function to create training and validation pairs holding the right amount of input data
    and the target slice

    Arguments:
        config (dict):  configuration dict holding hyperparameters
        data (List) :  List holding data from the various sets
    """

    if config['verbose']:
        print('Creating pairs')

    pairs_train, pairs_val = [], []

    for patient in data[0]:
        index_offset = int((config['N'] - 1) / 2)
        index_end = int(patient['source'].shape[0] - index_offset)
        for index in range(index_offset, index_end):
            sample = {
                # 'source': [s for s in patient['source'][int(index - index_offset): int(index + index_offset + 1)]],
                'source': np.expand_dims(patient['source'][index], 0),
                'target': np.expand_dims(patient['target'][index], 0)
            }
            pairs_train.append(sample)

    for patient in data[1]:
        index_offset = int((config['N'] - 1) / 2)
        index_end = int(patient['source'].shape[0] - index_offset)
        for index in range(index_offset, index_end):
            sample = {
                # 'source': [s for s in patient['source'][int(index - index_offset): int(index + index_offset)]],
                'source': np.expand_dims(patient['source'][index], 0),
                'target': np.expand_dims(patient['target'][index], 0)

            }
            pairs_val.append(sample)

    return pairs_train, pairs_val


def remove_empty_scans(config: dict, data: List, alpha: int = 30) -> List[List]:
    """
    Function to remove training and validation pairs that are empty

    Arguments:
        config (dict):  configuration dict holding hyperparameters
        data (List) :  List holding data from the various sets
        alpha (int) :  Cut-off value for removing empty slices
    """
    
    if config['verbose']:
        print('Removing empty slices')

    for set in data:
        for patient in set:
            target_list = list(patient['target'])
            source_list = list(patient['source'])
            indices_to_delete = []
            for index, scan in enumerate(target_list):
                if scan.sum() < alpha:
                    indices_to_delete.append(index)
            for index in sorted(indices_to_delete, reverse=True):
                del target_list[index]
                del source_list[index]
            patient['target'] = np.array(target_list)
            patient['source'] = np.array(source_list)

    return data
   