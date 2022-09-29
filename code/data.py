""""
File to hold the data related functions

"""

from torch.utils.data import Dataset, DataLoader

import numpy as np
import SimpleITK as sitk
import os
from typing import List


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
    padded = add_padding(config, data)
    train_pairs, val_pairs = create_pairs(config, padded)
    train_set = SR_Dataset(config, train_pairs)
    val_set = SR_Dataset(config, val_pairs)

    train_loader = DataLoader(train_set, batch_size=config['batchsize'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['batchsize'], shuffle=True, num_workers=0)

    return train_loader, val_loader


def read_files(config: dict) -> List[List]:
    """
    Function to read the files from the disk and organize them by type

    Arguments:
        config (dict): configuration dict holding hyperparameters

    """
    if config['verbose']:
        print('Loading data')

    os.chdir("..")
    os.chdir(config['path_train'])
    os.chdir('source_resolution')
    files = os.listdir()
    os.chdir('..')

    set_A = [f for f in files if ('SET_A' in f) and ('VSET' not in f)]
    set_B = [f for f in files if ('SET_B' in f) and ('VSET' not in f)]
    set_V = [f for f in files if 'VSET' in f]

    data = [[], [], []]
    sets = [set_A, set_B, set_V]

    for i in range(len(data)):
        for f in sets[i]:
            patient = {
                "source": sitk.GetArrayFromImage(sitk.ReadImage('source_resolution/' + f)),
                "target": sitk.GetArrayFromImage(sitk.ReadImage('target_resolution/' + f))
            }
            data[i].append(patient)

    if config['verbose']:
        print(
            f'''
        Found data:
            - set A: {len(data[0])}
            - set B: {len(data[1])}
            - set V: {len(data[2])}
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

    if config['N'] == 1:
        return data

    padding = int((config['N'] - 1) / 2)
    padded = [[], [], []]

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

    index_offset = int((config['N'] - 1) / 2)
    index_end = int(data[0][0]['source'].shape[0] - index_offset)

    for set in data[:-1]:
        for patient in set:
            for index in range(index_offset, index_end):
                sample = {
                    'source': [s for s in patient['source'][int(index - index_offset): int(index + index_offset + 1)]],
                    'target': patient['target'][index]
                }
                pairs_train.append(sample)

    for patient in data[-1]:
        for index in range(index_offset, index_end):
            sample = {
                'source': [s for s in patient['source'][int(index - index_offset): int(index + index_offset)]],
                'target': patient['target'][index]
            }
            pairs_val.append(sample)

    return pairs_train, pairs_val
