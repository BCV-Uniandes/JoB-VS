import os
import json
import shutil
import numpy as np
from joblib import Parallel, delayed

from utils import read_json, preprocess, cases_list


def Preprocess_datasets(out_dir, root, workers, remake):
    # ! Modified to read annotations in OASIS format. [N. Valderrama ~ May 17th]
    folds = ['fold1', 'fold2']
    main_root = '/media/SSD0/nfvalderrama/Vessel_Segmentation/data/tof'
    for x in folds:
        out_task = os.path.join(out_dir, x)
        os.makedirs(out_task, exist_ok=True)
        os.makedirs(os.path.join(out_task, 'imagesTr'), exist_ok=True)
        os.makedirs(os.path.join(out_task, 'imagesTs'), exist_ok=True)
        os.makedirs(os.path.join(out_task, 'labelsTr'), exist_ok=True)
        os.makedirs(os.path.join(out_task, 'labelsTs'), exist_ok=True)

        dataset, limits, stats, spacing, CT = read_json(x, root)
        dataset['root'] = os.path.join(out_dir, x)
        for split in ['training', 'validation']:
            partition = 'Tr' if split == 'training' else 'Ts'
            dataset[split] = [{'image': os.path.join(
                                        f'images{partition}', i['image']),
                                    'label': os.path.join(
                                        f'labels{partition}', i['label']),
                                    'monai_name': i['monai_name'],
                                    } for i in dataset[split]]
        affine = np.diag(spacing + [1])
        spacing = np.asarray(spacing)
        args = {'task': main_root, 'spacing': spacing,
                'limits': limits, 'stats': stats, 'path': out_task,
                'affine': affine, 'CT': CT}

        

        print('----- Processing training set -----')
        patientsTr = cases_list(dataset, out_task, 'imagesTr', 'training', remake=remake)
        Parallel(n_jobs=workers)(delayed(preprocess)(
            i['monai_name'], i['image'], args, lb=i['label']) for i in patientsTr)
        print('----- Processing test set -----')
        patientsTs = cases_list(dataset, out_task, 'imagesTs', 'validation', remake=remake)
        Parallel(n_jobs=workers)(delayed(preprocess)(
            i['monai_name'], i['image'], args, lb=i['label'], partition='Ts') for i in patientsTs)

        for split in ['training', 'validation']:
            partition = 'Tr' if split == 'training' else 'Ts'
            dataset[split] = [{'image': os.path.join(
                                        f'images{partition}',i['monai_name']),
                                    'label': os.path.join(
                                        f'labels{partition}', i['monai_name'])
                                } for i in dataset[split]]
        with open(os.path.join(out_task, 'dataset.json'), 'w') as file:
            json.dump(dataset, file, indent = 4)