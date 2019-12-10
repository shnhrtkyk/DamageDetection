#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:46:28 2019

@author: shino
"""

from shapely import wkt
import json
import pylab as plt
import numpy as np
from matplotlib.path import Path
from PIL import Image
import os 

width, height=1024, 1024

def polygon_area(x,y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5*np.abs(main_area + correction)

path = "/Volumes/Untitled/dmg/train/labels/"
outpath= "/Volumes/Untitled/dmg/train/masks/"
outpath_mini= "/Volumes/Untitled/dmg/train/masks_mini/"
outpath_mid= "/Volumes/Untitled/dmg/train/masks_mid/"
outpath_big= "/Volumes/Untitled/dmg/train/masks_big/"

files = os.listdir(path)

for filename in files:
    finalgt = np.zeros((width, height))
    finalgt_mini = np.zeros((width, height))
    finalgt_mid = np.zeros((width, height))
    finalgt_big = np.zeros((width, height))
    damage = 255
    if(filename[0] == "."):continue
    print(path + filename)
    
    file = open(path + filename, 'r')
#    json = json.loads(file.read())
    print(file)
    jsonfile = json.load(file)
    outname = outpath + filename.replace(".json",".jpg")
    outname_mini = outpath_mini + filename.replace(".json",".jpg")
    outname_mid = outpath_mid + filename.replace(".json",".jpg")
    outname_big = outpath_big+ filename.replace(".json",".jpg")
    print(outpath)
    file.close
    xlist = []
    ylist = []

    for i in jsonfile['features']['xy']:
        geom = wkt.loads(i['wkt'])
        pts = list(geom.exterior.coords)
        for j in range(len(pts)):

            xlist.append(pts[j][0])
            ylist.append(pts[j][1])
        if 'post' in filename:
            damage_cls =  i["properties"]["subtype"]
            if(damage_cls == "no-damage"):damage=1
            elif(damage_cls == "minor-damage"):damage=2
            elif(damage_cls == "major-damage"):damage=3
            elif(damage_cls == "destroyed"):damage=4
            else:damage=5
        poly_path=Path(pts)
    
        print(xlist)
        #print(pts)
        y, x = np.mgrid[:height, :width]
    
        coors = (np.hstack((x.reshape(-1, 1), y.reshape(-1,1))))# coors.shape is (4000000,2)

    
        mask = poly_path.contains_points(coors)
        area = polygon_area(np.array(xlist), np.array(ylist))
        print(area)
        finalgt = np.where(mask.reshape(height, width) == True,damage,finalgt)
    #    print(mask)
        if(area <= 100):#sizeの条件
            mask_mini = poly_path.contains_points(coors)
            finalgt_mini = np.where(mask_mini.reshape(height, width) == True,damage,finalgt_mini)
        elif(area > 100 and area <= 200):#sizeの条件
            mask_mid = poly_path.contains_points(coors)
            finalgt_mid = np.where(mask_mid.reshape(height, width) == True,damage,finalgt_mid)
        elif(area > 200):#sizeの条件
            mask_big = poly_path.contains_points(coors)
            finalgt_big = np.where(mask_big.reshape(height, width) == True,damage,finalgt_big)
    #    print(mask)
        
    #    print(pts)
    
    plt.imshow(finalgt.reshape(height, width))
    plt.show()




    #pilImg = Image.fromarray(np.uint8(finalgt))
    #pilImg.save(outname, 'JPEG')
    
    #pilImg = Image.fromarray(np.uint8(finalgt_mini))
    #pilImg.save(outname_mini, 'JPEG')
    
    #pilImg = Image.fromarray(np.uint8(finalgt_mid))
    #pilImg.save(outname_mid, 'JPEG')
    
    #pilImg = Image.fromarray(np.uint8(finalgt_big))
    #pilImg.save(outname_big, 'JPEG')

