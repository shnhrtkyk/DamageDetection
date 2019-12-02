#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          # 
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, # 
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

from PIL import Image
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import random
import argparse
import logging
import json
import cv2
import datetime

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict


def process_img(img_array, polygon_pts, scale_pct):
    """Process Raw Data into

            Args:
                img_array (numpy array): numpy representation of image.
                polygon_pts (array): corners of the building polygon.

            Returns:
                numpy array: extracted polygon image from img_array.

    """

    height, width, _ = img_array.shape

    #Find the four corners of the polygon
    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    #Extend image by scale percentage
    xmin = max(int(xmin - (xdiff * scale_pct)), 0)
    xmax = min(int(xmax + (xdiff * scale_pct)), width)
    ymin = max(int(ymin - (ydiff * scale_pct)), 0)
    ymax = min(int(ymax + (ydiff * scale_pct)), height)

    return img_array[ymin:ymax, xmin:xmax, :]


def process_img_poly(img_path_pre,img_path_post,  label_path,  output_dir_pre,output_dir_post, output_csv):
    x_data = [] 
    img_obj = Image.open(img_path_post)
    img_obj_pre = Image.open(img_path_pre)

    #Applies histogram equalization to image
    img_array = np.array(img_obj)
    img_array_pre = np.array(img_obj_pre)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #img_array = clahe.apply(img_array_pre)

    #Get corresponding label for the current image
    label_file = open(label_path)
    label_data = json.load(label_file)

    #Find all polygons in a given image
    for feat in label_data['features']['xy']:

        poly_uuid = feat['properties']['uid'] + ".png"

        # Extract the polygon from the points given
        polygon_geom = shapely.wkt.loads(feat['wkt'])
        polygon_pts = np.array(list(polygon_geom.exterior.coords))
        poly_img = process_img(img_array, polygon_pts, 0.8)
        poly_img_pre = process_img(img_array_pre, polygon_pts, 0.8)

        # Write out the polygon in its own image
        cv2.imwrite(output_dir_post + "/" + poly_uuid, poly_img)
        cv2.imwrite(output_dir_pre + "/" + poly_uuid, poly_img_pre)
        x_data.append(poly_uuid)

    data_array = {'uuid': x_data}
    df = pd.DataFrame(data = data_array)
    df.to_csv(output_csv)

def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--input_img_pre',
                        required=True,
                        metavar="/path/to/xBD_input_pre",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--input_img_post',
                        required=True,
                        metavar="/path/to/xBD_input_post",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--label_path',
                        required=True,
                        metavar="/path/to/xBD_input",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--output_dir_pre',
                        required=True,
                        metavar='/path/to/xBD_output_pre',
                        help="Path to new directory to save images")
    parser.add_argument('--output_dir_post',
                        required=True,
                        metavar='/path/to/xBD_output_post',
                        help="Path to new directory to save images")
    parser.add_argument('--output_csv',
                        required=True, 
                        metavar='/path/to/xBD_output',
                        help="Path to save the csv file")

    args = parser.parse_args()

    process_img_poly(args.input_img_pre, args.input_img_post, args.label_path, args.output_dir_pre, args.output_dir_post, args.output_csv)


if __name__ == '__main__':
    main()
