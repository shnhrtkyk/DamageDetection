from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from eval import eval_multiunet 
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append('/dd/code/pytorch-deeplab-xception/utils/')
from loss import SegmentationLosses

# -------- trainer-----------#
def train_net(model, optimizer,train_loader, val_loader, device, scheduler, epochs=5, loss_type='ce',  batch_size=16, lr=0.1, log_dir='log/', save_cp=True, use_class_weight=False, model_name='UNet'):
    logging.basicConfig(
            # filename=os.path.join(log_dir, 'train.log'), 
            level=logging.INFO, 
            format='%(levelname)s: %(message)s',
            handlers = [
                logging.FileHandler('{}/train.log'.format(log_dir)),
                logging.StreamHandler()
            ])

    logger = logging.getLogger()
    writer = SummaryWriter(log_dir=log_dir, comment='this is the start of training')
    global_step = 0
    n_train = train_loader.dataset.__len__()
    n_val = val_loader.dataset.__len__()
    dataset = train_loader.dataset
    logger.info(f'''Start training: 
            Epochs: {epochs}
            Batchsize: {batch_size}
            leraning rate: {lr}
            training_size: {n_train}
            validation_szie: {n_val}
            checkpoints: {save_cp}
            device: {device.type}
            ''')
    # define loss function
    if use_class_weight: 
        dataset = train_loader.dataset
        weight = dataset.get_class_weight() # numpy array
        weight = torch.from_numpy(weight.astype(np.float32))
    else:
        print('ok, dont use class weight. fair. ')
        weight = None 

    print('use class weight ! : {}'.format(weight))
    if device.type == 'cuda':
        flag = True
    else:
        flag = False

    # loss function for each size of buildings
    criterion = []
    for i in range(3):
        criterion.append(SegmentationLosses(weight=weight, ignore_index=2, cuda=flag).build_loss(mode=loss_type))
    best_pred = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        i = 0
        iter_count = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch[0]
                true_masks = batch[1]
                gt_binary = batch[2]
                imgs = imgs.to(device=device, dtype=torch.float32)
                for i in range(3):
                    true_masks[i] = true_masks[i].to(device=device, dtype=torch.long)

                masks_pred = model(imgs) # (batch_small, batch_medium, batch_large)
                loss = torch.Tensor([0]).cuda() 
                for j in range(3):
                    loss += criterion[j](masks_pred[j], true_masks[j])
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.update(imgs.shape[0])
                global_step += 1
                iter_count += 1
                # for debug
                if iter_count > 100: break

        if epoch % 5 == 0:
            val_score, new_pred = eval_multiunet(model, val_loader, device, n_val, logger, epoch)
            if new_pred > best_pred:
                best_pred = new_pred
            logger.info('Validation loss: {}, best fscore: {}'.format(val_score, best_pred))
            writer.add_scalar('Loss/test', val_score, global_step)
        writer.add_images('images', imgs, global_step)
        if save_cp:
            try:
                # os.mkdir(dir_checkpoint)
                model_dir = log_dir + 'models/'
                os.mkdir(model_dir)
                logger.info('Created checkpoint directory')
            except OSError:
                pass
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_dir + f'CP_epoch{epoch + 1}.pth')
            else:
                torch.save(model.state_dict(), model_dir + f'CP_epoch{epoch + 1}.pth')
            logger.info(f'Checkpoint {epoch + 1} saved !')


    writer.close()


