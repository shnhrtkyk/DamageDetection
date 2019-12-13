import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from dice_loss import dice_coeff
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import sys
sys.path.append('/dd/code/pytorch-deeplab-xception/utils/')
from loss import SegmentationLosses


def save_eval_output(save_dir, pred_final, gt, epoch):
    print('unique class in gt: {}'.format(np.unique(gt)))
    print('unique class in pred: {}'.format(np.unique(pred_final)))
    N = pred_final.shape[0]
    n = np.random.randint(N)
    pred2save = pred_final[n]
    gt2save = gt[n]
    cv2.imwrite(save_dir + 'pred_epoch_{}.png'.format(epoch), pred2save)
    cv2.imwrite(save_dir + 'gt_epoch_{}.png'.format(epoch), gt2save)


def filter_by_size(size, bin_img):
    new_image = np.zeros(bin_img.shape)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    for i, row in enumerate(stats):
        if i == 0:
            # skip this, becuz first one is object id 0, which is background.
            continue
        blank = np.zeros(new_image.shape)
        s_x, s_y = row[1], row[0]
        f_x, f_y = (row[1] + row[3]), (row[0] + row[2])
        valid_area = np.sqrt(np.sum(bin_img[s_x:f_x, s_y:f_y]))
        if len(size) == 2:
            if np.logical_and(valid_area >= size[0], valid_area < size[1]):
                blank[s_x:f_x, s_y:f_y] = bin_img[s_x:f_x, s_y:f_y]
                new_image += blank
        elif valid_area >= size[0]:
            blank[s_x:f_x, s_y:f_y] = bin_img[s_x:f_x, s_y:f_y]
            new_image += blank
    new_image = np.where(new_image >0, 1, 0) # filter the overlapped pixels
    return new_image

def filter_by_size_batch(size, bin_imgs):
    batchsize = bin_imgs.shape[0]
    new_images = []
    for i in range(batchsize):
        new_img = filter_by_size(size, bin_imgs[i])
        new_images.append(new_img)
    new_images_array = np.concatenate([new_images], 0)
    # print('inside filter by sioze batch: {}'.format(new_images_array.shape))
    # print(new_images_array.shape) # should be batch x size x size
    return new_images_array

def gen_bin_image(value, code, prob_data, prob_ind):
    '''
        value: value for truncate the probability
        code: which class to remain 
        prob_data: probability map
        prob_ind: predicted value 
    '''
    pred_truncated = np.where(prob_data > value, prob_ind, 0)
    pred_binarized = np.where(pred_truncated == code, pred_truncated, 0)
    return pred_binarized.astype(np.uint8) # for opencv connectedcomponents 


def eval_multiunet(net, loader, device, n_val, logger, epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    class_code = [1,2,3]
    thresholds = [0.5, 0.5, 0.5]
    size_limit = [[0, 25], [25, 50], [50]]
    tot = 0
    tot_tn = 0
    tot_fp = 0
    tot_fn = 0
    tot_tp = 0
    # criterion = nn.CrossEntropyLoss()
    criterion = []
    weight = None
    iter_count = 0
    for i in range(3):
        criterion.append(SegmentationLosses(weight=weight, ignore_index=2, cuda=True).build_loss(mode='ce'))

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0]
                true_masks = batch[1]
                gt_binary = batch[2]
                imgs = imgs.to(device=device, dtype=torch.float32)
                for i in range(3):
                    true_masks[i] = true_masks[i].to(device=device, dtype=torch.long)

                mask_pred = net(imgs) # (batch_small, batch_medium, batch_large) 
                loss = torch.Tensor([0]) 
                for j in range(3):
                    loss += criterion[j](mask_pred[j], true_masks[j])
                tot += loss 

                # 1) binarize, 2) remove by size, 3) OR, 4) remove too small 
                # scale wise 
                    # batch wise operation 
                pred_list = [] # length should be 3, number of scales 
                for k in range(3):
                    pred = F.softmax(mask_pred[k], dim=1)
                    pred = pred.permute(0, 2, 3, 1)
                    prob_data, prob_ind = pred.max(dim=-1)
                    prob_data, prob_ind = prob_data.cpu().numpy(), prob_ind.cpu().numpy()
                    bin_img = gen_bin_image(value=thresholds[k], code=class_code[k],prob_data=prob_data, prob_ind=prob_ind) 
                    filtered_bin_img = filter_by_size_batch(size=size_limit[k], bin_imgs=bin_img) 
                    # print(filtered_bin_img.shape) # should be batch x size x size
                    pred_list.append(filtered_bin_img)

                pred_integrate = sum(pred_list)  
                pred_integrate = np.where(pred_integrate > 0, 1, 0) # same as logical summation. 
                pred_final = filter_by_size_batch(size=[1], bin_imgs=pred_integrate.astype(np.uint8)) # filter out small noise

                gt = gt_binary.cpu().numpy()
                tn, fp, fn, tp = confusion_matrix(pred_final.reshape(-1), gt.reshape(-1)).ravel()
                tot_tn += tn
                tot_fp += fp
                tot_fn += fn
                tot_tp += tp
                # randomly save image
                pbar.update(imgs.shape[0])
                # for debug
                iter_count += 1
                if iter_count > 50: break

    # randomly save image
    save_eval_output('/dd/code/vislog/', pred_final, gt, epoch)
    precision = tot_tp / (tot_tp + tot_fp)
    recall = tot_tp / (tot_tp + tot_fn)
    fscore = 2 * precision * recall / (precision + recall)
    logger.info('val precision: {}'.format(precision))
    logger.info('val recall: {}'.format(recall))
    logger.info('val fscore: {}'.format(fscore))
    return tot / n_val, fscore
