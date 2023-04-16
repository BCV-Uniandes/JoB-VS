import os
import json
import shutil
import numpy as np
from joblib import Parallel, delayed

from utils import read_json, preprocess, cases_list


def Preprocess_datasets(out_dir, root, workers, remake):
    # ! Modified to read annotations in OASIS format. [N. Valderrama ~ May 17th]
    main_root = '/media/SSD0/nfvalderrama/Vessel_Segmentation/data/IXI_DATASET_2.0'
    out_task = os.path.join(out_dir)
    os.makedirs(out_task, exist_ok=True)
    os.makedirs(os.path.join(out_task, 'imagesTs'), exist_ok=True)

    dataset, limits, stats, spacing, CT = read_json(root)
    dataset['root'] = os.path.join(out_dir)
    for split in ['validation']:
        partition = 'Tr' if split == 'training' else 'Ts'
        dataset[split] = [{'image': os.path.join(
                                    f'images{partition}', i['image']),
                                'label': ''
                                } for i in dataset[split]]
    affine = np.diag(spacing + [1])
    spacing = np.asarray(spacing)
    args = {'task': main_root, 'spacing': spacing,
            'limits': limits, 'stats': stats, 'path': out_task,
            'affine': affine, 'CT': CT}

    
    print('----- Processing test set -----')
    patientsTs = cases_list(dataset, out_task, 'imagesTs', 'validation', remake=remake)
    Parallel(n_jobs=workers)(delayed(preprocess)(
        i['image'].split('/')[-1], i['image'], args, lb=None, partition='Ts') for i in patientsTs)

    with open(os.path.join(out_task, 'dataset.json'), 'w') as file:
        json.dump(dataset, file, indent = 4)