import os
import time
import torch
from options.test_options import TestOptions
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils.label2Img import label2rgb
from dataloader.transform import Transform_test
from dataloader.dataset import NeoData_test
from networks import get_model
from eval import *

def main(args):
    despath = args.savedir
    if not os.path.exists(despath):
        os.mkdir(despath)

    imagedir = os.path.join(args.datadir,'image.txt')     
    labeldir = os.path.join(args.datadir,'label.txt')            
                                         
    transform = Transform_test(args.size)
    dataset_test = NeoData_test(imagedir, labeldir, transform)
    loader = DataLoader(dataset_test, num_workers=4, batch_size=1,shuffle=False) #test data loader

    #eval the result of IoU
    confMatrix = evalIoU.generateMatrixTrainId(evalIoU.args)
    perImageStats = {}
    nbPixels = 0

    confMatrix_post = evalIoU.generateMatrixTrainId(evalIoU.args)
    perImageStats_post = {}
    nbPixels_post = 0
    usedLr_post = 0


    
    model = get_model(args)
    if args.cuda:
        model = model.cuda()
    model.load_state_dict(torch.load(args.model_dir))
    model.eval()
    count = 0
    for step, colign in enumerate(loader):
 
      img = colign[2].squeeze(0).numpy()       #image-numpy,original image   
      images = colign[0]                       #image-tensor
      label = colign[1]                        #label-tensor
      img_post = colign[5].squeeze(0).numpy()       #image-numpy_post,original image
      images_post = colign[3]                       #image_post-tensor
      label_post = colign[4]                        #label_post-tensor
      #images = images.transpose(1,2).trainspose(1,3).float 
      if args.cuda:
        images = images.cuda()
        images_post = images_post.cuda()

      inputs = Variable(images,volatile=True)
      inputs_post = Variable(images_post, volatile=True)

      outputs, outputs_post = model(inputs, inputs_post)
      out = outputs[0].cpu().max(0)[1].data.squeeze(0).byte().numpy() #index of max-channel
      out_post = outputs_post[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()  # index of max-channel
      
      add_to_confMatrix(outputs, label, confMatrix, perImageStats, nbPixels)  #add result to confusion matrix
      add_to_confMatrix(outputs_post, label_post, confMatrix_post, perImageStats_post, nbPixels_post)  # add result to confusion matrix
      print (label[0,:,:].cpu().data.numpy().shape)
      print (images[0,:,:].cpu().data.numpy().shape)
      label2img_gt = label2rgb(label[0,:,:].cpu().data.numpy())
      label2img = label2rgb(out)   #merge segmented result with original picture
      Image.fromarray(label2img).save(despath + 'B_' +str(count)+'_label2img.jpg' )
      #Image.fromarray(label2img_gt).save(despath + 'B_' +str(count)+'_lgt2img.jpg' )

      label2img_gt_post = label2rgb(label_post[0,:,:].cpu().data.numpy())
      label2img_post = label2rgb(out_post)   #merge segmented result with original picture
      Image.fromarray(label2img_post).save(despath + 'D_' +str(count)+'_label2img.jpg' )
      #Image.fromarray(label2img_gt_post).save(despath + 'D_' +str(count)+'_gt2img.jpg' )
      count += 1
      print("This is the {}th of image!".format(count))
        
    iouAvgStr, iouTest, classScoreList = cal_iou(evalIoU, confMatrix)  #calculate mIoU, classScoreList include IoU for each class
    iouAvgStr_post, iouTest_post, classScoreList_post = cal_iou(evalIoU,confMatrix_post)  # calculate mIoU, classScoreList include IoU for each class
    print("IoU on B set : ",iouAvgStr)
    print("IoU on D set : ",iouAvgStr_post)
    #print("IoU on TEST set of each class - car:{}  light:{} ".format(classScoreList['car'],classScoreList['light']))

if __name__ == '__main__':
    parser = TestOptions().parse()
    main(parser)


