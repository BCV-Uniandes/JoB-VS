import os
import csv
import torch
import matplotlib
import numpy as np
import pandas as pd
import os.path as osp

from scipy.stats import mode
from joblib import Parallel, delayed
from libs.utilities.test import read_image, dice_score
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib.pyplot as plt

matplotlib.pyplot.switch_backend('Agg')

EXPS_PATH = '/media/SSD0/nfvalderrama/Vessel_Segmentation/exps'

def draw_curve(scores, save_path):

    # Precision Recall curve
    f_max, rec, prec = scores[1], list(scores[-1]) + [0], list(scores[-2]) + [1]
    area = scores[0]
    plt.figure(figsize=(10, 5))
    plt.plot(rec, prec)
    plt.scatter(scores[2], scores[3], color="red") 
    plt.title(f'mAP {round(area*100, 3)}%. Mean max dice {round(f_max*100, 3)}% at threshold {scores[-3]} %')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(save_path + f'_PR_curve.png')
    plt.grid()
    
    plt.close('all')
    return scores
    
def parallel_test_ap(name, preds_dir, anns_dir):
    dice = []
    precision = []
    recall = []
    print(name)
    im_path = os.path.join(preds_dir, name)
    lb_path = os.path.join(anns_dir, name)

    im, affine = read_image(im_path)
    lb, _ = read_image(lb_path)
    lb = np.round(lb)

    y_true = lb.ravel()
    scores = im.ravel()
    ap = average_precision_score(y_true, scores)
    thresholds = np.linspace(0, 1, 51)
    for i in thresholds:
        im_thr = (im >= i).astype(np.uint8)
        dc, rec, prec = dice_score(im_thr, lb)
        dice.append(dc)
        recall.append(rec)
        precision.append(prec)
    
    max_dice_id = np.argmax(dice)
    max_dice = dice[max_dice_id]
    max_thr = thresholds[max_dice_id]
    max_prec = precision[max_dice_id]
    max_rec = recall[max_dice_id]
    
    return [name, ap, max_dice, max_prec, max_rec, max_thr, precision, recall]

def  write_in_folder(patients, path_preds):
    metrics = ['Mean AP', 'F-med max', 'Precision max', 'Recall max', 'Threshold', 'Precision', 'Recall']
    scores = []
    for i, metric in enumerate(metrics):
        score = [patient[i+1] for patient in patients]
        if metric == 'Threshold':
            score = mode(score).mode.item()
        else:
            score = np.asarray(score)
            score = np.mean(score, axis=0)
            
        scores.append(score)
    
    fields = ['Label'] + metrics[:5]
    
    with open(path_preds + '.csv', 'w') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=fields)
        writer.writeheader()

        for idx, j in enumerate(patients):
            line = {field: datum for field, datum in zip(fields, j)}
            writer.writerow(line)
            
        last = ['mean'] + scores[:5]
        writer.writerow({field: datum for field, datum in zip(fields, last)})
        outcsv.close()
    
    draw_curve(scores, path_preds)
    
    return scores[:5], {'precision': scores[-2], 'recall': scores[-1]}

def compute_metrics_ap(path_preds, path_anns, model, fold, inf_mode):
    path_preds = osp.join(EXPS_PATH, model, path_preds, fold, 'volumes', inf_mode)
    path_anns = osp.join(path_anns, fold, 'labelsTs')
    
    images = os.listdir(path_preds)
    print(f'-------Computing stats for {fold} {inf_mode} ------')
    patients = Parallel(n_jobs=10)(
            delayed(parallel_test_ap)(image, path_preds, path_anns)
            for image in images)
    
    metrics, curves = write_in_folder(patients, path_preds)
    return metrics, curves

if __name__ == '__main__':
    PATH_ANNS = '/media/SSD0/nfvalderrama/Vessel_Segmentation/data/Vessel_Segmentation/mask'
    PATH_PREDS = {
                  'JoB-VS': 'ROG',
                  }
    
    METRICS = ['Mean AP', 'F-med max', 'Precision max', 'Recall max', 'Threshold']
    
    save_path = osp.join(EXPS_PATH, 'Resume', 'vessel_seg_results')
    
    if os.path.isfile(save_path + '.pth'):
        pth_resume = torch.load(save_path + '.pth')
    else:
        pth_resume = []
        
    for pred, model in PATH_PREDS.items():
        for inf_mode in ['mask', 'original']:
            pd_resume = {}
            pd_resume['Name'] = [f'{model}_{pred}_{inf_mode}']
            for fold in ['fold2', 'fold1']:
                metrics, curves = compute_metrics_ap(pred, PATH_ANNS, model, fold, inf_mode)
                print(f'-------Saving stats for {fold} {inf_mode} ------')
                for i, metric in enumerate(metrics):
                    pd_resume[f'{METRICS[i]}_{fold}'] = [metric]
                pth_resume.append({f'{model}_{pred}_{inf_mode}_{fold}': curves})
    
            pd_resume = pd.DataFrame(pd_resume)
            pd_resume.to_csv(save_path + '.csv', mode='a', index=False, header=False)
            torch.save(pth_resume, save_path + '.pth')
    
    
    