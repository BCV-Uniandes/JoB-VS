# -*- coding: utf-8 -*-
from bdb import Breakpoint
import os
import time
import argparse
import numpy as np

import libs.trainer as trainer

from libs.model.model import ROG
from libs.model.unetr import UNETR
from settings import plan_experiment
from libs.dataloader import dataloader, helpers
from libs.utilities import losses, utils, test, cldice

import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.networks.nets import UNet

tasks = {'0': 'Vessel_Segmentation'}


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234' + port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    print(f"Running on rank {rank}.")
    setup(rank, world_size, args.port)

    training = args.test

    if args.ft:
        args.resume = True
        
    ver = 'mask' if args.mask else 'original'
    info, model_params = plan_experiment(
        tasks[args.task], args.batch,
        args.patience, args.fold,
        rank, args.model, args.data_ver ,ver)

    # PATHS AND DIRS
    args.save_path = os.path.join(
        info['output_folder'], args.name, f'fold{args.fold}')
    images_path = os.path.join(args.save_path, 'volumes')
    
    if args.ixi:
        images_path = os.path.join(args.save_path, 'IXI_DATASET')
    
    if args.our_masks:
        images_path = os.path.join(images_path, 'masks_pred')
    elif args.logits:
        images_path = os.path.join(images_path, ver)
    else:
        images_path = os.path.join(images_path, 'binary')

    load_path = args.save_path  # If we're resuming the training of a model
    if args.pretrained is not None:
        load_path = os.path.join(
            args.pretrained, f'fold{args.fold}')
    
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, 'brain'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels_brain'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels_logits'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'vessels_brain_logits'), exist_ok=True)

    # SEEDS
    np.random.seed(info['seed'])
    torch.manual_seed(info['seed'])

    cudnn.deterministic = False  # Normally is False
    cudnn.benchmark = args.benchmark  # Normaly is True

    # CREATE THE NETWORK ARCHITECTURE
    if args.model == 'ROG':
        model = ROG(model_params).to(rank)
    elif args.model == 'UNet':
        model = UNet(spatial_dims=len(model_params['img_size']),
                     in_channels=model_params['in_channels'],
                     out_channels=model_params['out_channels'],
                     channels=(4, 8, 16),
                     strides=(2, 2)).to(rank)
    else:
        print('Model not found')
        return
    
    ddp_model = DDP(model, device_ids=[rank])
    if rank == 0:
        f = open(os.path.join(args.save_path, 'architecture.txt'), 'w')
        print(model, file=f)
    scaler = amp.GradScaler()

    if training or args.ft:
        # Initialize optimizer
        optimizer = optim.Adam(
            ddp_model.parameters(), lr=args.lr,
            weight_decay=1e-5, amsgrad=True)
        annealing = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, patience=info['patience'], factor=0.5)
        # Save experiment description
        if rank == 0:
            name_d = 'description_train.txt'
            name_a = 'args_train.txt'
            if not training:
                name_d = 'description_test.txt'
                name_a = 'args_test.txt'

            with open(os.path.join(args.save_path, name_d), 'w') as f:
                for key in info:
                    print(key, ': ', info[key], file=f)
                for key in model_params:
                    print(key, ': ', model_params[key], file=f)
                print(
                    'Number of parameters:',
                    sum([p.data.nelement() for p in model.parameters()]),
                    file=f)

                with open(os.path.join(args.save_path, name_a), 'w') as f:
                    for arg in vars(args):
                        print(arg, ':', getattr(args, arg), file=f)

    # CHECKPOINT
    epoch = 0
    best_dice = 0
    
    if args.load_weights is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.load_weights, map_location=map_location)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        if 'rng' in checkpoint.keys():
            np.random.set_state(checkpoint['rng'][0])
            torch.set_rng_state(checkpoint['rng'][1])
        if 'module.' not in list(checkpoint.keys())[0]:
            checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
        model_dict = ddp_model.state_dict()
        # Match pre-trained weights that have same shape as current model.
        pre_train_dict_match = {
            k: v
            for k, v in checkpoint.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        # Weights that do not have match from the pre-trained model.
        not_load_layers = [
            k
            for k in model_dict.keys()
            if k not in pre_train_dict_match.keys()
        ]
        # Log weights that are not loaded with the pre-trained weights.
        if not_load_layers and rank==0:
            for k in not_load_layers:
                print("Network weights {} not loaded.".format(k))
        
        # Load pre-trained weights.
        ddp_model.load_state_dict(pre_train_dict_match, strict=False)
        
    if args.resume:
        name = args.load_model + '.pth.tar'
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(
            os.path.join(load_path, name),
            map_location=map_location)
        # Only for training. Must be loaded before loading the model
        if not args.ft:
            np.random.set_state(checkpoint['rng'][0])
            torch.set_rng_state(checkpoint['rng'][1])

        if rank == 0:
            print('Loading model epoch {}.'.format(checkpoint['epoch']))

        ddp_model.load_state_dict(
            checkpoint['state_dict'], strict=(not args.ft))
        # if ft, we do not need the previous optimizer
        if not args.ft:
            best_dice = checkpoint['best_dice']
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            annealing.load_state_dict(checkpoint['scheduler'])
        args.load_model = 'best_dice'

    criterion = losses.segmentation_loss(alpha=1)
    if args.cldice:
        criterion_vessels = cldice.soft_dice_cldice()
    else:
        criterion_vessels = None
    metrics_v = utils.Evaluator(info['classes'])
    metrics_b = utils.Evaluator(info['classes'])
    metrics = [metrics_v, metrics_b]

    # DATASETS
    train_dataset = dataloader.Medical_data(
        True, info['data_file'], info['root'], info['p_size'])
    val_dataset = dataloader.Medical_data(
        True, info['data_file'], info['root'], info['val_size'], val=True)
    test_dataset = dataloader.Medical_data(
        False, info['data_file'], info['root'], info['val_size'], val=True)

    # SAMPLERS
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    train_collate = helpers.collate(info['in_size'])
    val_collate = helpers.collate_val(list(map(int, info['val_size'])))

    # DATALOADERS
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=info['batch'],
        num_workers=8, collate_fn=train_collate)
    val_loader = DataLoader(
        val_dataset, sampler=None, batch_size=info['test_batch'],
        num_workers=8, collate_fn=val_collate)
    test_loader = DataLoader(
        test_dataset, sampler=None, shuffle=False, batch_size=1, num_workers=0)

    # TRAIN THE MODEL
    is_best = False
    torch.cuda.empty_cache()

    def moving_average(cum_loss, new_loss, n=5):
        if cum_loss is None:
            cum_loss = new_loss
        cum_loss = np.append(cum_loss, new_loss)
        if len(cum_loss) > n:
            cum_loss = cum_loss[1:]
        return cum_loss.mean()

    if training:
        accumulated_val_loss = None
        out_file = open(os.path.join(args.save_path, 'progress.csv'), 'a+')
        noise_data = torch.zeros(
            [info['batch'], model_params['modalities']] + info['in_size'],
            device=rank)
        for epoch in range(epoch + 1, args.epochs + 1):
            lr = utils.get_lr(optimizer)
            if rank == 0:
                print('--------- Starting Epoch {} --> {} ---------'.format(
                    epoch, time.strftime("%H:%M:%S")))
                print('Current learning rate:', lr)

            train_sampler.set_epoch(epoch)
            train_loss, noise_data = trainer.train(
                args, info, ddp_model, train_loader, noise_data, optimizer,
                criterion, scaler, rank, criterion_vessels)
            val_loss, dice = trainer.val(
                args, ddp_model, val_loader, criterion,
                metrics, rank, criterion_vessels)

            accumulated_val_loss = moving_average(
                accumulated_val_loss, val_loss)
            annealing.step(accumulated_val_loss)

            mean = sum(dice) / len(dice)
            is_best = best_dice < mean
            best_dice = max(best_dice, mean)

            # Save ckeckpoint (every 100 epochs, best model and last)
            if rank == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': ddp_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': annealing.state_dict(),
                    'rng': [np.random.get_state(),
                            torch.get_rng_state()],
                    'loss': [train_loss, val_loss],
                    'lr': lr,
                    'dice': dice,
                    'best_dice': best_dice}
                checkpoint = epoch % 100 == 0
                utils.save_epoch(
                    state, mean, args.save_path, out_file,
                    checkpoint=checkpoint, is_best=is_best)

            if lr <= (args.lr / (2 ** 4)):
                print('Stopping training: learning rate is too small')
                break
        out_file.close()

    # Loading the best model for testing
    dist.barrier()
    torch.cuda.empty_cache()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    name = args.load_model + '.pth.tar'
    checkpoint = torch.load(
        os.path.join(args.save_path, name), map_location=map_location)
    torch.set_rng_state(checkpoint['rng'][1]) 
    ddp_model.load_state_dict(checkpoint['state_dict'])
    if rank == 0:
        print('Testing epoch with best dice ({}: dice {})'.format(
            checkpoint['epoch'], checkpoint['dice']))

    # EVALUATE THE MODEL
    trainer.test(args,
            info, ddp_model, test_loader, images_path,
            info['data_file'], rank, world_size)
    dist.barrier()
    # CALCULATE THE FINAL METRICS
    classes = 3 if args.detection else 4
    if rank == 0:
        test.test(
            folder=images_path, root_dir=info['root'], 
            csv_file=info['data_file'], classes=classes,
            detection=args.detection)
    cleanup()


if __name__ == '__main__':
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    # EXPERIMENT DETAILS
    parser.add_argument('--task', type=str, default='0',
                        help='Task to train/evaluate (default: 4)')
    parser.add_argument('--model', type=str, default='ROG',
                        help='Model to train with the ROG training curriculum (default: ROG)')
    parser.add_argument('--data_ver', type=str, default='/media/SSD0/nfvalderrama/Vessel_Segmentation/data/Vessel_Segmentation/',
                        help='Path to data')
    parser.add_argument('--name', type=str, default='ROG',
                        help='Name of the current experiment (default: ROG)')
    parser.add_argument('--AT', action='store_true', default=False,
                        help='Train a model with Free AT')
    parser.add_argument('--fold', type=str, default=1,
                        help='Which fold to run. Value from 1 to 2')
    parser.add_argument('--test', action='store_false', default=True,
                        help='Evaluate a model')
    parser.add_argument('--aux_train', action='store_true', default=False,
                        help='Sample more patches per patient')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Continue training a model')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='Fine-tune a model (will not load the optimizer)')
    parser.add_argument('--load_model', type=str, default='best_dice',
                        help='Weights to load (default: best_dice)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Name of the folder with the pretrained model')

    # TRAINING HYPERPARAMETERS
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs (default: 1000)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience of the scheduler (default: 50)')
    parser.add_argument('--batch', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='Path to load initial weights (default: None)')
    
    # ADVERSARIAL TRAINING AND TESTING
    parser.add_argument('--eps', type=float, default=8.,
                        help='Epsilon for the adv. attack (default: 8/255)')
    parser.add_argument('--alpha_vessels', type=float, default=0.5,
                        help='Multiplication factor in vessels loss'),
    parser.add_argument('--alpha_brain', type=float, default=0.5,
                        help='Multiplication factor in brain loss'),
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU(s) to use (default: 0)')
    parser.add_argument('--port', type=str, default='5')
    parser.add_argument('--benchmark', action='store_false', default=True,
                        help='Deactivate CUDNN benchmark')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='Use data with the brain masks')
    parser.add_argument('--cldice', action='store_true', default=False,
                        help='Use cldice for vessel segmentation')
    parser.add_argument('--our_masks', action='store_true', default=False,
                        help='Use data with the brain masks predicted by our models')
    parser.add_argument('--detection', action='store_true', default=False,
                        help='Evaluate with detection metrics')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold to evaluate the predictions (default: None)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main, args=(world_size, args,), nprocs=world_size, join=True)
    else:
        # To allow breakpoints
        main(0, 1, args)
