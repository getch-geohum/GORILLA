
import random
import numpy as np
from glob import glob
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils import remapLC_filter, remamDGlob, onehot, toFloat, train_testsplit
from skimage.transform import resize

class LandDataset(Dataset):
    def __init__(self, imgs, lbls, n_classes, shape, subset=True, one_encoding=False, sub_type='random'):
        self.imgs = imgs
        self.lbls = lbls
        self.shape = shape
        self.n_classes = n_classes
        self.subset = subset
        self.sub_type = sub_type
        self.one_encoding = one_encoding
        
        self.tile_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.inds = self.tile_indices[0]
        
        self.min = np.array([0.0, 0.0, 0.0, 0.0]).reshape(1,1,4)
        self.max = np.array([0.65, 0.65, 0.65, 0.65]).reshape(1,1,4)

        self.mus = np.array([0.040551468815817705, 0.05671991797227009, 0.05664462136485029, 0.1688796614838653]).reshape(1,1,4)
        self.stds = np.array([0.04558734906216333, 0.0473301615731358,  0.051660152632441746, 0.12516235881460264]).reshape(1,1,4)

        assert len(self.imgs) == len(self.lbls), f'images and labels numbers {len(self.imgs)} and {len(self.lbls)} respectively are not the same'

    def __len__(self):
        return len(self.imgs)
    
    def set_tileindex(self, idx_val):
       assert idx_val<=3, 'as tile index has four values, its index cannot be greater than 3'
       self.inds = self.tile_indices[idx_val]

    def process_image(self, image):
        img = imread(image)
        if self.subset:
          # if self.sub_type == 'random':
          #   self.inds = random.choice(self.tile_indices)
          # else:
          #   self.inds = self.tile_indices[0]
          ri, ci = self.inds
          img = img[ri*256:ri*256+256, ci*256:ci*256+256,:]
        else:
          img = img[:self.shape, :self.shape, :]

        img = (img-img.min())/(img.max() - img.min() + 0.000001)
        # img = np.clip(img, 0, 0.65) # 0.67
        # img = (img-self.min)/(self.max - self.min + 0.000001)
        # img = (img-self.mus)/(self.stds)
        return img

    def process_label(self, label):
        lbl = imread(label)

        if self.subset:
          ri, ci = self.inds
          lbl = lbl[ri*256:ri*256+256, ci*256:ci*256+256]
        else:
          lbl = lbl[:self.shape, :self.shape]

        # lbl = np.resize(lbl, new_shape=(self.shape, self.shape, 4)
        lbl = remapLC_filter(lbl) # needs checkups
        if self.one_encoding:
            return onehot(lbl, self.n_classes)
        else:
            return lbl-1
        
    def __getitem__(self, idx):
        imfile = self.imgs[idx]
        lblfile = self.lbls[idx]
        
        p_img = self.process_image(imfile)
        p_lbl = self.process_label(lblfile)
      

        dim_c = len(p_lbl.shape) == 3
        
        return torch.from_numpy(p_img).permute(2,0,1), torch.from_numpy(p_lbl).permute(2,0,1).float() if dim_c else torch.from_numpy(p_lbl).float()
    

class DigitalGlobdataset(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs # train_testsplit(files=imgs, part=part) if sample else imgs
        self.lbls = lbls #train_testsplit(files=lbls, part=part) if sample else lbls
        
        
        assert len(self.imgs) == len(self.lbls), f'In dataset the number of images and labels {len(self.imgs)} and {len(self.lbls)} respectively are not the same'

    def __len__(self):
        return len(self.imgs)

    def process_image(self, image):
        img = imread(image)
    
        img = img/255 # (img-img.min())/(img.max() - img.min() + 0.000001)
        return img

    def process_label(self, label):
        lbl = imread(label)
        return lbl
        
    def __getitem__(self, idx):
        imfile = self.imgs[idx]
        lblfile = self.lbls[idx]
        
        p_img = self.process_image(imfile)
        p_lbl = self.process_label(lblfile)

        dim_c = len(p_lbl.shape) == 3

        
        return torch.from_numpy(p_img).permute(2,0,1), torch.from_numpy(p_lbl).permute(2,0,1).float() if dim_c else torch.from_numpy(p_lbl).long()

