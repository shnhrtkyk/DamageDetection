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



path = "/Volumes/Untitled/train/labels/"
outpath= "/Volumes/Untitled/train/masks/"
files = os.listdir(path)

for filename in files:
    finalgt = np.zeros((width, height))
    damage = 255
    if(filename[0] == "."):continue
    print(path + filename)
    
    file = open(path + filename, 'r')
#    json = json.loads(file.read())
    print(file)
    jsonfile = json.load(file)
    outname = outpath + filename.replace(".json",".jpg")
    print(outpath)
    file.close

    for i in jsonfile['features']['xy']:
        geom = wkt.loads(i['wkt'])
        pts = list(geom.exterior.coords)
        if 'post' in filename:
            damage_cls =  i["properties"]["subtype"]
            if(damage_cls == "no-damage"):damage=1
            elif(damage_cls == "minor-damage"):damage=2
            elif(damage_cls == "major-damage"):damage=3
            elif(damage_cls == "destroyed"):damage=4
            else:damage=5
        poly_path=Path(pts)
    
    #    print(poly_path)
        y, x = np.mgrid[:height, :width]
    
        coors = (np.hstack((x.reshape(-1, 1), y.reshape(-1,1))))# coors.shape is (4000000,2)
    
        mask = poly_path.contains_points(coors)
    #    print(mask)
        finalgt = np.where(mask.reshape(height, width) == True,damage,finalgt)
    #    print(pts)
    
    plt.imshow(finalgt.reshape(height, width))
    plt.show()


    pilImg = Image.fromarray(np.uint8(finalgt))
    pilImg.save(outname, 'JPEG')

