import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png','.JPG','.PNG']


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def image_path(root, basename, extension):
    return os.path.join(root, '{}{}'.format(basename,extension))


def image_path_city(root, name):
    return os.path.join(root, '{}'.format(name))


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class NeoData(Dataset):
    def __init__(self, imagepath=None, labelpath=None, transform=None):
        #  make sure label match with image 
        self.transform = transform 
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        self.image = []
        self.label= []
        self.image_post = []
        self.label_post= []
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip())
        with open(imagepath,'r') as f:
            for line in f:
                self.image_post.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label_post.append(line.strip())

    def __getitem__(self, index):
        filename = "/home/elsa/shinohara/DamageDetection/data/patch/input_pre/" + self.image[index]
        filenameGt = "/home/elsa/shinohara/DamageDetection/data/patch/label_b/" + self.label[index]
        filename_post = "/home/elsa/shinohara/DamageDetection/data/patch/input_post/" + self.image[index]
        filenameGt_post = "/home/elsa/shinohara/DamageDetection/data/patch/label_d/" + self.label[index]

        with open(filename, 'rb') as f: 
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')
            label = np.asarray(label).astype("i")
            label = np.where(label >= 1 , 1, 0)
            label = Image.fromarray(np.uint8(label))
        with open(filename_post, 'rb') as f:
            image_post = load_image(f).convert('RGB')
        with open(filenameGt_post, 'rb') as f:
            label_post = load_image(f).convert('P')
            label_post = np.asarray(label_post).astype("i")
            label_post = np.where(label_post < 0, 0, label_post)
            label_post = np.where(label_post > 1, 2, label_post)
            #label_post = np.where(label_post == 150, 2, label_post)
            #label_post = np.where(label_post == 200, 3, label_post)
            #label_post = np.where(label_post == 255, 4, label_post)
            #label_post = np.where(label_post > 4, 4, label_post)
            label_post = Image.fromarray(np.uint8(label_post))
        if self.transform is not None:
            image, label = self.transform(image, label)
            image_post, label_post = self.transform(image_post, label_post)
        return image, label,image_post,label_post

    def __len__(self):
        return len(self.image)
    
class NeoData_test(Dataset):
    def __init__(self, imagepath=None, labelpath=None, transform=None):
        self.transform = transform 
        
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        self.image = []
        self.label= []
        self.image_post = []
        self.label_post= []
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip())
        with open(imagepath,'r') as f:
            for line in f:
                self.image_post.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label_post.append(line.strip())
        print("Length of test data is {}".format(len(self.image)))

    def __getitem__(self, index):
        filename = "/home/elsa/shinohara/DamageDetection/data/patch/input_pre/" + self.image[index]
        filenameGt = "/home/elsa/shinohara/DamageDetection/data/patch/label_b/" + self.label[index]
        filename_post = "/home/elsa/shinohara/DamageDetection/data/patch/input_post/" + self.image[index]
        filenameGt_post = "/home/elsa/shinohara/DamageDetection/data/patch/label_d/" + self.label[index]

        with open(filename, 'rb') as f: # advance
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')
            label = np.asarray(label).astype("i")
            label = np.where(label >= 1 , 1, 0)
            label = Image.fromarray(np.uint8(label))
        with open(filename_post, 'rb') as f:
            image_post = load_image(f).convert('RGB')
        with open(filenameGt_post, 'rb') as f:
            label_post = load_image(f).convert('P')
            label_post = np.asarray(label_post).astype("i")
            label_post = np.where(label_post < 0, 0, label_post)
            #label_post = np.where(label_post == 100, 1, label_post)
            #label_post = np.where(label_post == 150, 2, label_post)
            #label_post = np.where(label_post == 200, 3, label_post)
            #label_post = np.where(label_post == 255, 4, label_post)
            label_post = np.where(label_post > 4, 0, label_post)
            label_post = Image.fromarray(np.uint8(label_post))
        if self.transform is not None:
            image_tensor, label_tensor, img = self.transform(image, label)
            image_tensor_post, label_tensor_post, img_post = self.transform(image_post, label_post)

        return (image_tensor, label_tensor, np.array(img), image_tensor_post, label_tensor_post, np.array(img_post) ) #return original image, in order to show segmented area in origin

    def __len__(self):
        return len(self.image)

