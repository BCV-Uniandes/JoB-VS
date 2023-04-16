import os
import json
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed


def read_json(task, data_root):
    with open(os.path.join(data_root, task, 'dataset.json'), 'r') as f:
        dataset = json.load(f)
        numTraining = dataset['numTraining']
        CT = dataset['modality']['0'] == 'CT'

    print('Procesing task {}: training {}'.format(task, numTraining))
    return dataset, CT


def dataset_features(root, i, labels, CT):
    image = os.path.join(root, i['image'])
    print(image)
    im = nib.load(image)
    spacing = im.header.get_zooms()[:3]
    try:
        im = im.get_fdata()
    except:
        print('ESTA NO FUNCIONA', image)
    # If MRI, rescale the intensity range between 0 and 1
    if not CT:
        im = (im - im.min()) / (im.max() - im.min())
    # Select some pixels to calculate the statistics (1 every 10)
    pixels = []
    for cat in range(1, labels):
        pixels.append(im[im==cat][::10])
    return [pixels, spacing]


def global_stats(mean, std, size):
    total = np.sum(size, axis=0)
    mean_total = sum(mean * size) / total
    mean_var = ((size - 1) * std ** 2) + (size * (mean - mean_total) ** 2)
    mean_var = (mean_var).sum() / (total - 1)
    return mean_total, np.sqrt(mean_var)


def Global_features(root, num_workers):
    # ! Modified to read annotations in OASIS format. [N. Valderrama ~ May 17th]
    file = os.path.join(root, f'stats.json' )
    if os.path.isfile(file):
        with open(file, 'r') as f:
            stats_file = json.load(f)
            print('Statistics already calculated')
            statistics = stats_file
            return
    print('Calculating stats')
    
    json_file = os.path.join(root, f'data.json')
    dataset = json.load(open(json_file, 'r'))
    CT = dataset['modality']['0'] == 'CT'
    labels = len(dataset['labels'])
    features = Parallel(n_jobs=num_workers)(delayed(dataset_features)(
        dataset['root'], j, labels, CT) for j in dataset['validation'])
    values, spacing = [list(x) for x in zip(*features)]
    modalities = [list(x) for x in zip(*values)]

    low = np.ones(len(dataset['modality']))
    high = np.zeros(len(dataset['modality']))
    for modality in modalities:
        cat_values = np.concatenate(modality)
        low = np.minimum(low, np.percentile(cat_values, 0.5, axis=0))
        high = np.maximum(high, np.percentile(cat_values, 99.5, axis=0))

    mean, std, size = [], [], []
    for patient in values:
        p_values = np.concatenate(patient)
        if len(low) > 1:
            mask = []
            for ch in range(len(low)):
                mask.append(np.logical_and(p_values[:, ch] > low[ch],
                                            p_values[:, ch] < high[ch]))
            mask = np.stack(mask, axis=1)
        else:
            mask = np.logical_and(p_values > low, p_values < high)
        mean.append(np.mean(p_values * mask, axis=0))
        std.append(np.std(p_values * mask, axis=0))
        size.append(np.sum(mask, axis=0))
    mean, std, size = np.array(mean), np.array(std), np.asarray(size)
    mean, std = global_stats(mean, std, size)
    spacing = np.median(np.asarray(spacing), axis=0).tolist()

    info = {'case': 1, 'low': low, 'high': high}
    statistics = {
        'spacing': spacing,
        'mean': mean.tolist(),
        'std': std.tolist(),
        '0.5': np.squeeze(low).tolist(),
        '99.5': np.squeeze(high).tolist()}
    write_file = os.path.join(root, 'stats.json')
    with open(write_file, 'w') as outfile:
        json.dump(statistics, outfile, indent=4)
