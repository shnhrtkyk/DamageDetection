from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from eval import eval_net

# -------- trainer-----------#
def train_net(model, optimizer, train_loader, val_loader, device, epochs=5, batch_size=16, lr=0.1, log_dir='log/', save_cp=True):
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO, format='%(levelname)s: %(message)s')
    writer = SummaryWriter(log_dir=log_dir, comment='this is the start of training')
    global_step = 0
    n_train = train_loader.dataset.__len__()
    n_val = val_loader.dataset.__len__()
    dataset = train_loader.dataset
    logging.info(f'''Start training: 
            Epochs: {epochs}
            Batchsize: {batch_size}
            leraning rate: {lr}
            training_size: {n_train}
            validation_szie: {n_val}
            checkpoints: {save_cp}
            device: {device.type}
            ''')
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch[0]
                true_masks = batch[1]
                assert imgs.shape[1] == model.n_channels
                # assert true_masks.shape[1] == model.n_classes
                imgs = imgs.to(device=device, dtype=torch.float32)
                # true_masks = true_masks.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                masks_pred = model(imgs)
                print(masks_pred.shape)
                print(true_masks.shape)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(model, val_loader, device, n_val)
                    if model.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                    writer.add_images('images', imgs, global_step)
                    if model.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        if save_cp:
            try:
                # os.mkdir(dir_checkpoint)
                model_dir = log_dir + 'models/'
                os.mkdir(model_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(), model_dir + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


