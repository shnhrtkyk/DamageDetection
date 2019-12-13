#!/usr/bin/env python

import os
import numpy as np
import random
from tqdm import tqdm 
import cv2
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import six

# from chainer.dataset import dataset_mixin
from torch.utils.data import Dataset

# from transforms import random_color_distort


def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))


def _read_label_image_as_array(path, dtype):
    f = Image.open(path)
    f = f.convert('1')
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image

def _read_label_image_as_array_cv2(path):
    try:
        image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    except:
        print('somehow image cannot be read')
    return image



def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image

# pytorch dataset
class LabeledImageDataset_multiclass(Dataset):
    def __init__(self, dataset, root, label_root, dtype=np.float32,
                 label_dtype=np.int32, mean=0, crop_size=256, test=False,
                 distort=False):
        _check_pillow_availability()
        if isinstance(dataset, six.string_types):
            dataset_path = dataset
            with open(dataset_path) as f:
                pairs = []
                for i, line in enumerate(f):
                    line = line.rstrip('\n')
                    image_filename = line
                    label_filename = line
                    pairs.append((image_filename, label_filename))
        self._pairs = pairs
        self._root = root
        self._label_root = label_root
        self._dtype = dtype
        self._label_dtype = label_dtype
        self._mean = mean[np.newaxis, np.newaxis, :]
        self._crop_size = crop_size
        self._test = test
        self._distort = distort

    def __len__(self):
        return len(self._pairs)

    def get_class_weight(self):
        class_weight_dir = '/dd/code/class_weight/'
        try: 
           ret = np.load(class_weight_dir + 'class_weight_multi.npy') 
           print('the weight is pre-computed, good. (multi-class one)')
        except:
            print('ok, the weight is not comptuted yet (multi-class one.) ...')
            num_classes = 4 
            count = np.zeros((num_classes, ))
            # compute frequency 
            print('calculating class weights ... ')
            for i in tqdm(range(len(self._pairs))):
                label_filename = self._pairs[i][1]
                label_path = os.path.join(self._label_root, label_filename)
                # label_image = _read_label_image_as_array(label_path, self._label_dtype)
                label_image = _read_label_image_as_array_cv2(label_path)
                # h, w, _ = label_image.shape
                # label = np.zeros(shape=[h, w], dtype=np.int32) # 0: background
                count[0] += np.sum(label_image == 0) 
                count[1] += np.sum(label_image == 1)
                count[2] += np.sum(label_image == 2)
                count[3] += np.sum(label_image == 3)
            # compute weight
            total = np.sum(count)
            class_weights = []
            for frequency in count:
                class_weight = 1 / (np.log(1.02 + (frequency / total))) 
                class_weights.append(class_weight)
            ret = np.asarray(class_weights)
            class_weight_dir = '/dd/code/class_weight/'
            if not os.path.exists(class_weight_dir):
                os.makedirs(class_weight_dir)
            np.save(class_weight_dir + 'class_weight_multi.npy', ret)
        print('class weight: {}'.format(ret))
        return ret 

    def __getitem__(self, i):
        image_filename, label_filename = self._pairs[i]
        
        image_path = os.path.join(self._root, image_filename)
        image = _read_image_as_array(image_path, self._dtype)


        if self._distort:
            image = random_color_distort(image)
            image = np.asarray(image, dtype=self._dtype)

        image = (image - self._mean) / 255.0
        
        label_path = os.path.join(self._label_root, label_filename)
        # label_image = _read_label_image_as_array(label_path, self._label_dtype)
        label_image = _read_label_image_as_array_cv2(label_path)
        # print(np.unique(label_image))
        
        h, w, _ = image.shape
        
        label = label_image.copy()
        # gt = label_image.copy()
        # gt[label_image > 0] = 1
        # Padding
        if (h < self._crop_size) or (w < self._crop_size):
            H, W = max(h, self._crop_size), max(w, self._crop_size)
            
            pad_y1, pad_x1 = (H - h) // 2, (W - w) // 2
            pad_y2, pad_x2 = (H - h - pad_y1), (W - w - pad_x1)
            image = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')

            if self._test:
                # Pad with ignore_value for test set
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'constant', constant_values=255)
            else:
                # Pad with original label for train set  
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'symmetric')
            
            h, w = H, W
        
        # Randomly flip and crop the image/label for train-set
        if not self._test:

            # Horizontal flip
            if random.randint(0, 1):
                image = image[:, ::-1, :]
                label = label[:, ::-1]

            # Vertical flip
            if random.randint(0, 1):
                image = image[::-1, :, :]
                label = label[::-1, :]                
            
            # Random crop
            while True:
                top  = random.randint(0, h - self._crop_size)
                left = random.randint(0, w - self._crop_size)
                bottom = top + self._crop_size
                right = left + self._crop_size
                temp = label[top:bottom, left:right]
                if np.sum(temp > 0) != 0:
                    break

        # Crop the center for test-set
        else:
            top = (h - self._crop_size) // 2
            left = (w - self._crop_size) // 2
        
        bottom = top + self._crop_size
        right = left + self._crop_size
        
        image = image[top:bottom, left:right]
        label = label[top:bottom, left:right]
        label_1 = self.gen_binary_img(1, label)
        label_2 = self.gen_binary_img(2, label)
        label_3 = self.gen_binary_img(3, label)
        gt = label.copy()
        gt[label > 0] = 1 # all buildings are 1
        return image.transpose(2, 0, 1).copy(), (label_1.copy(),label_2.copy(),label_3.copy()), gt

    def gen_binary_img(self, code, label):
        '''
          generate binary image that contains:
            0: background
            1: specified building size label
            2: other buildings
          
          input: ndarray image (1024, 1024)
          output:  ndarray image
        '''
        size = label.shape
        new_label = np.zeros(size)
        new_label[label == code] = 1
        new_label[label > code] = 2 # labels to ignore in practice
        new_label[np.logical_and(label > 0, label < code)] = 2 # labels to ignore in practice 
        return new_label 


class LabeledImageDataset(Dataset):
    def __init__(self, dataset, root, label_root, dtype=np.float32,
                 label_dtype=np.int32, mean=0, crop_size=256, test=False,
                 distort=False):
        _check_pillow_availability()
        if isinstance(dataset, six.string_types):
            dataset_path = dataset
            with open(dataset_path) as f:
                pairs = []
                for i, line in enumerate(f):
                    line = line.rstrip('\n')
                    image_filename = line
                    label_filename = line
                    pairs.append((image_filename, label_filename))
        self._pairs = pairs
        self._root = root
        self._label_root = label_root
        self._dtype = dtype
        self._label_dtype = label_dtype
        self._mean = mean[np.newaxis, np.newaxis, :]
        self._crop_size = crop_size
        self._test = test
        self._distort = distort

    def __len__(self):
        return len(self._pairs)

    def get_class_weight(self):
        class_weight_dir = '/dd/code/class_weight/'
        try: 
           ret = np.load(class_weight_dir + 'class_weight.npy') 
           print('the weight is pre-computed, good.')
        except:
            print('ok, the weight is not comptuted yet ...')
            num_classes = 2
            count = np.zeros((num_classes, ))
            # compute frequency 
            print('calculating class weights ... ')
            for i in tqdm(range(len(self._pairs))):
                label_filename = self._pairs[i][1]
                label_path = os.path.join(self._label_root, label_filename)
                label_image = _read_label_image_as_array(label_path, self._label_dtype)
                # h, w, _ = label_image.shape
                # label = np.zeros(shape=[h, w], dtype=np.int32) # 0: background
                count[1] += np.sum(label_image > 0)
                count[0] += np.sum(label_image == 0) 
            # compute weight
            total = np.sum(count)
            class_weights = []
            for frequency in count:
                class_weight = 1 / (np.log(1.02 + (frequency / total))) 
                class_weights.append(class_weight)
            ret = np.asarray(class_weights)
            class_weight_dir = '/dd/code/class_weight/'
            if not os.path.exists(class_weight_dir):
                os.makedirs(class_weight_dir)
            np.save(class_weight_dir + 'class_weight.npy', ret)
        print('class weight: {}'.format(ret))
        return ret 

    def __getitem__(self, i):
        image_filename, label_filename = self._pairs[i]
        
        image_path = os.path.join(self._root, image_filename)
        image = _read_image_as_array(image_path, self._dtype)


        if self._distort:
            image = random_color_distort(image)
            image = np.asarray(image, dtype=self._dtype)

        image = (image - self._mean) / 255.0
        
        label_path = os.path.join(self._label_root, label_filename)
        label_image = _read_label_image_as_array(label_path, self._label_dtype)
        
        h, w, _ = image.shape

        label = np.zeros(shape=[h, w], dtype=np.int32) # 0: background
        label[label_image > 0] = 1 # 1: "building"
        
        # Padding
        if (h < self._crop_size) or (w < self._crop_size):
            H, W = max(h, self._crop_size), max(w, self._crop_size)
            
            pad_y1, pad_x1 = (H - h) // 2, (W - w) // 2
            pad_y2, pad_x2 = (H - h - pad_y1), (W - w - pad_x1)
            image = np.pad(image, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'symmetric')

            if self._test:
                # Pad with ignore_value for test set
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'constant', constant_values=255)
            else:
                # Pad with original label for train set  
                label = np.pad(label, ((pad_y1, pad_y2), (pad_x1, pad_x2)), 'symmetric')
            
            h, w = H, W
        
        # Randomly flip and crop the image/label for train-set
        if not self._test:

            # Horizontal flip
            if random.randint(0, 1):
                image = image[:, ::-1, :]
                label = label[:, ::-1]

            # Vertical flip
            if random.randint(0, 1):
                image = image[::-1, :, :]
                label = label[::-1, :]                
            
            # Random crop
            top  = random.randint(0, h - self._crop_size)
            left = random.randint(0, w - self._crop_size)
        
        # Crop the center for test-set
        else:
            top = (h - self._crop_size) // 2
            left = (w - self._crop_size) // 2
        
        bottom = top + self._crop_size
        right = left + self._crop_size
        
        image = image[top:bottom, left:right]
        label = label[top:bottom, left:right]
        # transpose ?      
        # return image.transpose(2, 0, 1).copy(), np.expand_dims(label, 0).copy()
        return image.transpose(2, 0, 1).copy(), label.copy()

