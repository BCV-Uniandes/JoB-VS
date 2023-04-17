# -*- coding: utf-8 -*-
import os
import math
import json
import numpy as np
import torch.nn as nn

# batch sizes used for training the network
precalculated_batch = {
    '0': 1, # [192, 192, 96]
    '1': 3,  # [128, 128, 96]
    '2': 2,  # [192, 192, 96]
    '3': 2,  # [192, 192, 96]
    '4': 22,  # [32, 48, 32]
    '5': 13,  # [96, 96, 16]
    '6': 2,  # [192, 192, 96]
    '7': 3,  # [192, 128, 64]
    '8': 5,  # [192, 128, 42]
    '9': 5,  # [192, 192, 48]
    '10': 3,  # [192, 192, 64]
    '11': 'Task11_KiTS'
}


def plan_experiment(root, task, batch, patience, fold, rank):
    root = '/'.join(root.split('/')[:-1])
    with open(os.path.join(root, 'dataset_stats.json'), 'r') as f:
        dataset = json.load(f)
        mean_size = dataset['mean_size']
        small_size = dataset['small_size']
        volume = dataset['volume_small']
        modalities = len(dataset['modality'])
        classes = len(dataset['labels'])

    # Asuming the structures are cubes, if the volume of a structure is
    # smaller than 32^3 we should use only 4 downsamples
    threshold = 32 * 32 * 32
    # breakpoint()
    # ! Num_downsampes can be removed as we don't need to automatize according 
    # ! to the task.
    num_downsamples = 4 if volume < threshold else 5

    # num_downsamples = 3
    # Analysis per axis
    tr_size, val_size, p_size, strides, size_last = calculate_sizes(
        num_downsamples, mean_size, small_size)

    # MEMORY CONSTRAINT 1
    # If the feature maps in the final stage are too big we make sure to use
    # five downsamples
    if size_last >= (12 * 12 * 6):
        tr_size, val_size, p_size, strides, size_last = calculate_sizes(
            5, mean_size, small_size)
        num_downsamples = 5

    # MEMORY CONSTRAINT 2
    # If the feature maps in the final stage are still too big
    # we reduce the input size
    if size_last >= (12 * 12 * 4):
        tr_size, val_size, p_size, strides, size_last = calculate_sizes(
            num_downsamples, mean_size, small_size, 6)
    # Define the architecture topology
    tr_size = [128, 128, 128]
    val_size = tr_size
    p_size = tr_size
    # strides = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    strides = list(map(list, zip(*strides)))
    if len(strides) == 4:
        strides.insert(0, [1, 1, 1])

    if rank == 0:
        print('Current task is {}, with {} modalities and {} classes'.format(
            task, modalities, classes))
        print('The mean size of the images in this dataset is:', mean_size)
        print('--- Training input size set to {} ---'.format(tr_size))
        print('--- Validation input size set to {} ---'.format(val_size))

    hyperparams = {
        'task': task,
        'classes': classes,
        'p_size': p_size,  # size of the patch before the transformations
        'in_size': tr_size,  # size of the patch that enters the network
        'val_size': val_size,  # size of the patch that enters the network
        'test_size': val_size,  # size of the patch that enters the network
        'batch': batch,
        'test_batch': int(batch * 1),
        'patience': patience,  # Make it dependent on the data?
        'seed': 12345,
        'output_folder': '',
        'root': os.path.join(root, f'fold{fold}'),
        'data_file': os.path.join(root, f'fold{fold}', 'dataset.json'),
        # 'val_file': os.path.join('Tasks', task, f'val_fold{fold}.csv'),
        # 'test_file': os.path.join('Tasks', task, f'val_fold{fold}.csv')
        # 'test_file': os.path.join('Tasks', task, 'test_paths.csv')
    }

    model = {
        'classes': classes,
        'modalities': modalities,
        'strides': strides[:3],
        'img_size': tr_size,
        'in_channels': modalities,
        'out_channels': classes,
        'feature_size': 48,
        'use_checkpoint': True,
    }
    return hyperparams, model


def calculate_sizes(num_downsamples, mean_size, small_size, max_pow=7):
    tr_size = []
    val_size = []
    strides = []
    p_size = []
    size_last = 1
    # Analysis per axis
    for i, j in zip(mean_size, small_size):
        # If the image is too big we'll treat it as if it was smaller to avoid
        # ending up with huge input patches (memory constraint)
        i_big = i
        if i > 128:
            i_big = i
            i *= 0.7

        # Calculate the maximum possible input patch size
        power_t = min(int(math.log(i, 2)), max_pow)  # Max 64 or 128
        power_v = min(int(math.log(i, 2)), 7)  # Max 128
        sz_t = pow(2, power_t)
        sz_v = pow(2, power_v)

        # Calculate the number of strides
        stride = min(min(int(math.log(j, 2)), num_downsamples), power_t - 2)
        temp = np.ones(num_downsamples, dtype=int) * 2
        temp[:-stride] = 1
        strides.append(list(temp))

        # Calculate the input patch size
        constraint = sz_t / (2 ** stride) >= 8 and num_downsamples == 5
        if sz_t * 1.5 < i and not constraint:
            sz_t *= 1.5  # Max 196
        if sz_v * 1.5 < i:
            sz_v *= 1.5  # Max 196
        tr_size.append(int(sz_t))
        val_size.append(np.maximum(int(i_big // 2), int(sz_v)))
        size_last *= (sz_t / (2 ** stride))

        if sz_t + 20 < i:
            p_size.append(int(sz_t + 20))
        else:
            p_size.append(int(sz_t))
    return tr_size, val_size, p_size, strides, size_last
