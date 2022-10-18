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

# SET_A_MAX = 0
# SET_B_MAX = 0
# SET_C_MAX = 0

# SET_A_MAX_V = 0
# SET_B_MAX_V = 0
# SET_C_MAX_V = 0


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

def normalize_values(f: string):
    """
    Function to normalize the data values between 0 and 1

    Arguments:
        f (string): filename of image file

    """
    # global SET_A_MAX, SET_B_MAX, SET_C_MAX, SET_A_MAX_V, SET_B_MAX_V, SET_C_MAX_V

    source = sitk.GetArrayFromImage(sitk.ReadImage('source_resolution/' + f))
    target = sitk.GetArrayFromImage(sitk.ReadImage('target_resolution/' + f))

    if 'SET_A' in f:
        if 'VSET' in f:
            # if source.max() > SET_A_MAX_V:
            #     SET_A_MAX_V = source.max()
            source = source / 1936
            target = target / 1936
        else:
            # if source.max() > SET_A_MAX:
            #     SET_A_MAX = source.max()
            source = source / 557.88635
            target = target / 557.88635
    if 'SET_B' in f:
        if 'VSET' in f:
            # if source.max() > SET_B_MAX_V:
            #     SET_B_MAX_V = source.max()
            source = source / 6207
            target = target / 6207
        else:
            # if source.max() > SET_B_MAX:
            #     SET_B_MAX = source.max()
            source = source / 2250.5593
            target = target / 2250.5593
        # source = source / 6207
        # target = target / 6207
    if 'SET_C' in f:
        if 'VSET' in f:
            # if source.max() > SET_C_MAX_V:
            #     SET_C_MAX_V = source.max()
            source = source / 17274
            target = target / 17274
        else:
            # if source.max() > SET_C_MAX:
            #     SET_C_MAX = source.max()
            source = source / 0
            target = target / 0
        # source = source / 17274
        # target = target / 17274

    # print(SET_A_MAX, SET_B_MAX, SET_C_MAX, SET_A_MAX_V, SET_B_MAX_V, SET_C_MAX_V)

    # source = source / 255
    # target = target / 255

    return source, target


def read_files(config: dict) -> List[List]:
    """
    Function to read the files from the disk and organize them by type

    Arguments:
        config (dict): configuration dict holding hyperparameters

    """
    if config['verbose']:
        print('Loading data')

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

    if int(config['N']) == 1:
        return data

    padding = int((int(config['N']) - 1) / 2)
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

    for set in data[:-1]:
        for patient in set:
            index_offset = int((config['N'] - 1) / 2)
            index_end = int(patient['source'].shape[0] - index_offset)
            for index in range(index_offset, index_end):
                sample = {
                    # 'source': [s for s in patient['source'][int(index - index_offset): int(index + index_offset + 1)]],
                    'source': np.expand_dims(patient['source'][index], 0),
                    'target': np.expand_dims(patient['target'][index], 0)
                }
                pairs_train.append(sample)

    for patient in data[-1]:
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
   