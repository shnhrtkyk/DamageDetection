#!/usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import multiprocessing
# from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from trainer_multi_unet import train_net
from dataset_xiu import LabeledImageDataset_multiclass
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('/dd/code/pytorch-deeplab-xception/')
#sys.path.append('/dd/code/unet/')
# from modeling.deeplab import *
# from utils.lr_scheduler import LR_Scheduler

def train_model():
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', help='Path to directory containing train.txt, val.txt, and mean.npy')
    parser.add_argument('images',  help='Root directory of input images')
    parser.add_argument('labels',  help='Root directory of label images')
    
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--batch_size_val', '-B', type=int, default=4,
                        help='Number of images in each test mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=str, default='as much as possible',
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the result under "models" directory')
    parser.add_argument('--pretrained', '-p', type=str, default='',
                        help='use pretrained model')
    parser.add_argument('--tcrop', type=int, default=400,
                        help='Crop size for train-set images')
    parser.add_argument('--vcrop', type=int, default=400,
                        help='Crop size for validation-set images')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--model_name', type=str, default='UNet',
                        help='which model to use ')
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='which loss function to use [ce, focal]')
    parser.add_argument('--use_class_weight', type=int, default=0,
                        help='if we use class weights in loss function')
    parser.add_argument('--backbone', type=str, default='',
                        help='backbone netork for deeplabv3+')

    args = parser.parse_args()

    assert (args.tcrop % 16 == 0) and (args.vcrop % 16 == 0), "tcrop and vcrop must be divisible by 16."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print('# GPU: {}'.format(args.gpu))
    print('# device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# Crop-size: {}'.format(args.tcrop))
    print('# epoch: {}'.format(args.epochs))
    print('# class weight: {}'.format(args.use_class_weight))
    print('')
    
    # this_dir = os.path.dirname(os.path.abspath(__file__))
    # models_dir = os.path.normpath(os.path.join(this_dir, "models"))
    # if not os.path.exists(model_dir):
    #     os.mkdir(model_dir)
    log_dir = args.out
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Load mean image
    mean = np.load(os.path.join(args.dataset, "mean.npy"))
    
    # set dataset and dataloader
    train = LabeledImageDataset_multiclass(os.path.join(args.dataset, "train.txt"), args.images, args.labels, 
                                mean=mean, crop_size=args.tcrop, test=False, distort=False)
    
    val = LabeledImageDataset_multiclass(os.path.join(args.dataset, "val.txt"), args.images, args.labels, 
                                mean=mean, crop_size=args.vcrop, test=True, distort=False)
    num_workers = multiprocessing.cpu_count()
    # num_workers = 14 
    print('use {} workers !'.format(num_workers))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) 
    val_loader = DataLoader(val, batch_size=args.batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=True)

    if args.model_name == 'multi_unet':
        sys.path.append('/dd/code/unet/')
        from multi_scale_unet import multiunet 
        model = multiunet(n_channels=3, n_classes=4)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999)) 
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print('use {} GPUs ! '.format(torch.cuda.device_count()))
            model = model.cuda()
            model = nn.DataParallel(model)
        else:
            print('use only 1 GPU !')
            model = model.cuda()

    # TODO: load pre-trained model 

    

   # training 
    train_net(model=model, optimizer=optimizer, epochs=args.epochs, scheduler=scheduler, loss_type=args.loss_type, train_loader=train_loader, val_loader=val_loader, batch_size=args.batch_size,lr=args.lr, log_dir=args.out, device=device, use_class_weight=args.use_class_weight, model_name=args.model_name)






if __name__ == '__main__':
    train_model()
