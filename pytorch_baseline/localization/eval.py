import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from dice_loss import dice_coeff
from sklearn.metrics import confusion_matrix
# from 
def eval_net(net, loader, device, n_val, logger):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    tot_tn = 0
    tot_fp = 0
    tot_fn = 0
    tot_tp = 0
    criterion = nn.CrossEntropyLoss()
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0]
                true_masks = batch[1]

                imgs = imgs.to(device=device, dtype=torch.float32)
                # true_masks = true_masks.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)# (batch, size, size)

                mask_pred = net(imgs)# (batch, n_class, size, size)
                tot += criterion(mask_pred, true_masks)
                pred = F.log_softmax(mask_pred, dim=1)
                pred = pred.permute(0,2,3,1)
                pred = pred.data.max(-1)[1]
                # print(pred.shape) # (batch x size x size)
                pred = pred.cpu().numpy() 
                gt = true_masks.cpu().numpy()
                tn, fp, fn, tp = confusion_matrix(pred.reshape(-1), gt.reshape(-1)).ravel()
                tot_tn += tn
                tot_fp += fp
                tot_fn += fn
                tot_tp += tp
                pbar.update(imgs.shape[0])

                # for true_mask in true_masks:
                #     mask_pred = (mask_pred > 0.5).float()
                #     if net.n_classes > 1:
                #         # tot += F.cross_entropy(mask_pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                #         tot += criterion(mask_pred, true_mask).item()
                #     else:
                #         tot += dice_coeff(mask_pred, true_mask.squeeze(dim=1)).item()
                # pbar.update(imgs.shape[0])
    precision = tot_tp / (tot_tp + tot_fp)
    recall = tot_tp / (tot_tp + tot_fn)
    fscore = 2 * precision * recall / (precision + recall)
    logger.info('val precision: {}'.format(precision))
    logger.info('val recall: {}'.format(recall))
    logger.info('val fscore: {}'.format(fscore))
    return tot / n_val
