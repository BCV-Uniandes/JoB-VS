# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

import libs.utilities.utils as utils
import libs.dataloader.helpers as helpers

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

import nibabel as nib

def padding(prediction,pad, shape):
    return prediction[pad[0][0]:shape[0] - pad[0][1],
                      pad[1][0]:shape[1] - pad[1][1],
                      pad[2][0]:shape[2] - pad[2][1]]
    
def save_figs(in_data, output, ann):
    spacing = [0.2994791567325592, 0.2994791567325592, 0.5999984741210938]
    affine = np.diag(spacing + [1])
    in_data = in_data.detach().cpu().numpy()
    output = torch.argmax(torch.softmax(output, dim = 1), dim = 1).cpu().numpy()
    ann = ann.cpu().numpy()
    for i in range(len(in_data)):
        
        new_data = nib.Nifti1Image(in_data[i][0], affine)
        nib.save(new_data, f'prueba/prueba_in_{i}.nii.gz')
        new_data = nib.Nifti1Image(output[i], affine)
        nib.save(new_data, f'prueba/prueba_out_{i}.nii.gz')
        new_data = nib.Nifti1Image(ann[i], affine)
        nib.save(new_data, f'prueba/prueba_ann_{i}.nii.gz')

def get_aux(loader, idxs, rank):
    data = []
    target = []
    b_target = []
    for idx in idxs:
        samp = loader.dataset.sample_center_and_corners(idx)
        
    samp = loader.collate_fn([samp])
    aux_data = samp['data'].float().to(rank)
    aux_target = samp['target'].squeeze_(1).long().to(rank)
    aux_brain_target = samp['brain_target'].squeeze_(1).long().to(rank)
            
    return aux_data, aux_target, aux_brain_target

def train(args, info, model, loader, noise_data, optimizer, criterion, scaler,
          rank, criterion_vessels=None):
    model.train()
    loader.dataset.change_epoch()
    epoch_loss = utils.AverageMeter()
    batch_loss = utils.AverageMeter()

    iterations = args.adv_iters if args.AT else 1
    print_freq = max(1, len(loader) // 2)
    eps = args.eps / 255.
    
    for batch_idx, sample in enumerate(loader):
        data = sample['data'].float().to(rank)

        # Rescale the eps! (important for Free AT)
        b_min = torch.amin(data, [2, 3, 4], keepdim=True)
        b_max = torch.amax(data, [2, 3, 4], keepdim=True)
        b_eps = (b_max - b_min) * eps

        target = sample['target'].squeeze_(1).long().to(rank)
        brain_target = sample['brain_target'].squeeze_(1).long().to(rank)
        if args.aux_train:
            aux_data, aux_target, aux_brain_target = get_aux(loader, sample['idx'], rank)
        
        for _ in range(iterations):
            optimizer.zero_grad()
            in_data = data
            if args.AT:
                delta = noise_data[0:data.size(0)].to(rank)
                delta.requires_grad = True
                # ! Clamp
                in_data = torch.min(torch.max(data + delta, b_min), b_max) 

            with amp.autocast():
                    
                out = model(in_data)
                if criterion_vessels is None:
                    loss_v = criterion(out[0], target)
                else:
                    loss_v = criterion_vessels(target, out[0])
                    
                loss_b = criterion(out[1], brain_target)
                loss = args.alpha_brain*(loss_b) + args.alpha_vessels*(loss_v)
                
                scaler.scale(loss).backward()
                
                if args.aux_train:
                    aux_out = model(in_data)
                    loss_v = criterion(aux_out[0], aux_target)
                    loss_b = criterion(aux_out[1], aux_brain_target)
                    loss = loss_b + loss_v

                    scaler.scale(loss).backward()

            if args.AT:
                # Update the adversarial noise
                grad = delta.grad.detach()
                noise_data[0:data.size(0)] += (
                    (b_eps // 2) * torch.sign(grad)).data
                # ! Clamp
                noise_data[0:data.size(0)] =  torch.min(torch.max(
                    noise_data[0:data.size(0)], -b_eps), b_eps)

            scaler.step(optimizer)
            scaler.update()
            batch_loss.update(loss.item())
            epoch_loss.update(loss.item())

        if batch_loss.count % print_freq == 0:
            if rank == 0:
                text = '{} -- [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(text.format(
                    time.strftime("%H:%M:%S"), (batch_idx + 1),
                    (len(loader)), 100. * (batch_idx + 1) / (len(loader)),
                    batch_loss.avg))
            batch_loss.reset()
    if rank == 0:
        print('--- Train: \tLoss: {:.6f} ---'.format(epoch_loss.avg))
    return epoch_loss.avg, noise_data


def val(args, model, loader, criterion, metrics, rank, criterion_vessels=None):
    model.eval()
    metrics_v, metrics_b = metrics
    metrics_v.reset()
    metrics_b.reset()
    epoch_loss = utils.AverageMeter()

    for _, sample in enumerate(loader):
        data = sample['data'].float().to(rank)
        target = sample['target'].squeeze_(1).long()
        brain_target = sample['brain_target'].squeeze_(1).long()

        with torch.no_grad():
            out = model(data)
        if criterion_vessels is None:
            loss_v = criterion(out[0], target.to(rank))
        else:
            loss_v = criterion_vessels(target.to(rank), out[0])
        loss_b = criterion(out[1], brain_target.to(rank))
        loss = args.alpha_brain*(loss_b) + args.alpha_vessels*(loss_v)

        predictions = []
        for out_item in out:
            prediction = F.softmax(out_item, dim=1)
            prediction = torch.argmax(prediction, dim=1).cpu().numpy()
            predictions.append(prediction)
            
        metrics_v.add_batch(target.numpy(), predictions[0])
        metrics_b.add_batch(brain_target.numpy(), predictions[1])

        epoch_loss.update(loss.item(), n=target.shape[0])
    dice_v = metrics_v.Dice_Score()
    dice_b = metrics_b.Dice_Score()
    mean_dice = (dice_v + dice_b) / 2
    if rank == 0:
        print('--- Val: \tLoss: {:.6f} \tVessels Dice fg: {} \tBrain Dice fg: {} \tMean Dice fg: {} ---'.format(
            epoch_loss.avg, dice_v, dice_b, mean_dice))
    return epoch_loss.avg, mean_dice


def test(info, model, loader, images_path, test_file, rank, world_size):
    '''
    The inference is done by uniformly extracting patches of the images.
    The patches migth overlap, so we perform a weigthed average based on
    the distance of each voxel to the center of their corresponding patch.
    '''
    # ! Modified to read annotations in UNETR format
    # ! May 25th by N. Valderrama.
    patients = helpers.convert_format(test_file, val=True)
    model.eval()

    # Add more weight to the central voxels
    w_patch = np.zeros(info['val_size'])
    sigmas = np.asarray(info['val_size']) // 8
    center = torch.Tensor(info['val_size']) // 2
    w_patch[tuple(center.long())] = 1
    w_patch = gaussian_filter(w_patch, sigmas, 0, mode='constant', cval=0)
    w_patch = torch.Tensor(w_patch / w_patch.max()).to(rank).half()
    
    for idx in range(rank, len(patients), world_size):
        shape, name, affine, pad = loader.dataset.update(idx)
        prediction = torch.zeros((info['classes'],) + shape).to(rank).half()
        brain_prediction = torch.zeros((info['classes'],) + shape).to(rank).half()
        weights = torch.zeros(shape).to(rank).half()

        for sample in loader:
            data = sample['data'].float()  # .squeeze_(0)
            with torch.no_grad():
                output = model(data.to(rank))
            # output = dataloader.test_data(output, False)
            output[0][0] *= w_patch
            output[1][0] *= w_patch

            low = (sample['target'][0] - center).long()
            up = (sample['target'][0] + center).long()
            prediction[:, low[0]:up[0], low[1]:up[1], low[2]:up[2]] += output[0][0]
            brain_prediction[:, low[0]:up[0], low[1]:up[1], low[2]:up[2]] += output[1][0]
            weights[low[0]:up[0], low[1]:up[1], low[2]:up[2]] += w_patch

        # Vessels Prediction
        prediction /= weights
        prediction = F.softmax(prediction, dim=0)
        prediction_logits = torch.nan_to_num(prediction[1]).cpu()
        prediction = torch.argmax(prediction, dim=0).cpu()
        
        # Brain Prediction
        brain_prediction /= weights
        brain_prediction = F.softmax(brain_prediction, dim=0)
        brain_prediction = torch.argmax(brain_prediction, dim=0).cpu()
        
        if pad is not None:
            prediction = padding(prediction, pad, shape)
            prediction_logits = padding(prediction_logits, pad, shape)
            brain_prediction = padding(brain_prediction, pad, shape)

        # Save argmax predictions
        helpers.save_image(
            prediction, os.path.join(images_path, 'vessels', name), affine)
        helpers.save_image(
            brain_prediction, os.path.join(images_path, 'brain', name), affine)
        helpers.save_image(
            brain_prediction*prediction, 
             os.path.join(images_path, 'vessels_brain', name), affine)
        
        # Save softmax predictions
        helpers.save_image(
            prediction_logits.double(), os.path.join(images_path, 'vessels_logits', name), affine)
        helpers.save_image(
            (prediction_logits*brain_prediction).double(), 
             os.path.join(images_path, 'vessels_brain_logits', name), affine)
        
        print('Prediction {} saved'.format(name))

