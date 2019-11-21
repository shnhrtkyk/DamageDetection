# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:33:04 2019

@author: 006403
"""


import os 
from PIL import Image,ImageOps
import numpy as np
import scipy.stats as stats
import pandas as pd

pathroot = "D:/DamageDetection/train/"
child_dir_b = pathroot +  "building/train/"
child_dir_d = pathroot +  "damage/train/"
input_dir_pre = pathroot + "pre/train/"
input_dir_post = pathroot + "post/train/"

saveinput_pre = pathroot + "sample/input_pre/"
saveinput_post = pathroot + "sample/input_post/"
savegt_b =  pathroot +"sample/label_b/"
savegt_d =  pathroot +"sample/label_d/"

count = 0
STEP=256
SIZE=256
PATCH=256

count=0
for file_name in os.listdir(input_dir_pre):
    if(file_name[len(file_name)-3:] !="png"):continue
    print (file_name)
    imgpath_pre  = input_dir_pre + file_name
    imgpath_post  = input_dir_post + file_name.replace("pre","post")
    gtpath_b  = child_dir_b + file_name.replace("png","jpg")     
    gtpath_d  = child_dir_d + file_name.replace("pre","post").replace("png","jpg")   

    gt_b = Image.open(gtpath_b) 
    # flip 
    gt_b = ImageOps.mirror(gt_b)
    gt_b = ImageOps.mirror(gt_b)
    
    gt_b = np.asarray(gt_b).astype("i")
    gt_b = np.where(gt_b > 128, 1, 0)
    gt_b = Image.fromarray(np.uint8(gt_b))
    
    gt_d = Image.open(gtpath_d)
    gt_d = np.asarray(gt_d).astype("i")
    gt_d = np.where(gt_d < 1, 0, gt_d)
    gt_d = np.where(gt_d > 5, 0, gt_d)
    labels = np.unique(gt_d[gt_d >= 0]) 
    print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))
    gt_d = Image.fromarray(np.uint8(gt_d))   
    
    ori_pre = Image.open(imgpath_pre)       
    ori_post = Image.open(imgpath_post)                                                                                                                                                    
    maxwidth, maxheight = ori_pre.size
    XSIZE = 0
    YSIZE = 0
    
    
    for i in range(100000):
        for j in range(100000):
            
            XSIZE = i*STEP
            YSIZE = j*STEP
            if(XSIZE >maxwidth or YSIZE >maxheight):break
#            print(XSIZE,YSIZE,PATCH+XSIZE, PATCH+YSIZE)
            ori_crp_pre = ori_pre.crop((XSIZE, YSIZE, XSIZE+PATCH, YSIZE+PATCH))            
            ori_crp_post = ori_post.crop((XSIZE, YSIZE, XSIZE+PATCH, YSIZE+PATCH))
            gt_crp_b = gt_b.crop((XSIZE, YSIZE, XSIZE+PATCH, YSIZE+PATCH))
            gt_crp_d = gt_d.crop((XSIZE, YSIZE, XSIZE+PATCH, YSIZE+PATCH))
            s_zero = str(count).zfill(8)
            savepath_pre = saveinput_pre + file_name.replace(".png","_"+str(s_zero)+".jpg") 
            savepath_post = saveinput_post + file_name.replace(".png","_"+str(s_zero)+".jpg") 
            
            
            savepath_gt_b = savegt_b + file_name.replace(".png","_"+str(s_zero)+".jpg") 
            savepath_gt_d = savegt_d + file_name.replace(".png","_"+str(s_zero)+".jpg") 
                   
            gt_crp_b.save(savepath_gt_b)
            gt_crp_d.save(savepath_gt_d)
            ori_crp_pre.save(savepath_pre)
            ori_crp_post.save(savepath_post)
            
            count+=1