#!/usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
from unet import UNet
from att_unet import AttU_Net
# from eval import eval_net
import torch
import torch.nn as nn
import torch.optim as optim
from trainer import train_net
from dataset_xiu import LabeledImageDataset
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import os


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
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the result under "models" directory')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--tcrop', type=int, default=400,
                        help='Crop size for train-set images')
    parser.add_argument('--vcrop', type=int, default=480,
                        help='Crop size for validation-set images')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='learning rate')
    parser.add_argument('--model', type=str, default="unet",
                        help='select model: unet or att_unet')

    args = parser.parse_args()

    assert (args.tcrop % 16 == 0) and (args.vcrop % 16 == 0), "tcrop and vcrop must be divisible by 16."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# Crop-size: {}'.format(args.tcrop))
    print('# epoch: {}'.format(args.epochs))
    print('')
    
    # this_dir = os.path.dirname(os.path.abspath(__file__))
    # models_dir = os.path.normpath(os.path.join(this_dir, "models"))
    # if not os.path.exists(model_dir):
    #     os.mkdir(model_dir)
    log_dir = args.out
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if(args.model == unet):model = UNet(n_channels=3, n_classes=1) # for one class and background, n_classes=1
    elif(args.model == att_unet):model = AttU_Net(n_channels=3, n_classes=1) # for one class and background, n_classes=1

    if args.gpu >= 0:
        model = model.cuda()


    lr = args.lr # hyper param
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) 
    
    # Load mean image
    mean = np.load(os.path.join(args.dataset, "mean.npy"))
    
    # set dataset and dataloader
    train = LabeledImageDataset(os.path.join(args.dataset, "train.txt"), args.images, args.labels, 
                                mean=mean, crop_size=args.tcrop, test=False, distort=False)
    
    val = LabeledImageDataset (os.path.join(args.dataset, "val.txt"), args.images, args.labels, 
                                mean=mean, crop_size=args.vcrop, test=True, distort=False)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=7, pin_memory=True) 
    val_loader = DataLoader(val, batch_size=args.batch_size_val, shuffle=False, num_workers=7, pin_memory=True)


   # training 
    train_net(model=model, optimizer=optimizer, epochs=args.epochs, train_loader=train_loader, val_loader=val_loader, batch_size=args.batch_size,lr=args.lr, log_dir=args.out, device=device)






if __name__ == '__main__':
    train_model()
