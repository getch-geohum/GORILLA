import cv2
import os
import glob
from numpy.lib import imag
from skimage.io import imread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from utils import remapLC_filter

img_dir = "/root/.cache/kagglehub/datasets/getachewworkineh/uganda-landcover/versions/4/landcover_data_v2"  # Enter Directory of all images


def get_class_frequency(files, n_class=11):
    # n_train = len(loader)

    class_freq = np.zeros(n_class, dtype=float)

    for j in tqdm(range(len(files))):
        img = imread(files[j])
        img = remapLC_filter(img)
        (unique, counts) = np.unique(img, return_counts=True)

        for idx, i in enumerate(unique.tolist()):
            class_freq[i-1] += counts[idx]
            
    return class_freq


folds = ['train', 'test', 'valid']
for fold in folds:
  img_path = f'{img_dir}/{fold}'
  files = glob(f'{img_path}/labels/*.tif')
  print(f'Total {fold} data: ', len(files))
  freqencies = get_class_frequency(files=files,n_class=8)
  class_names = ["Tree cover",
                          "Shrubland", 
                          'Grassland',
                          'Cropland',
                          'Built-up', 
                          'Bare/sparse vegetation', 
                          'Permanent water bodies', 
                          'Herbaceous wetland' 
                          ]
  tot = freqencies.sum()
  print('===========================================')
  for i, clss in enumerate(class_names):
      percent = round((freqencies[i]/tot)*100, 3)
      print(f"Cover type {clss}: {percent}")
  print('===========================================')


  df = pd.DataFrame({'Classes':class_names,
                    'Samples':freqencies})

  ax = df.plot.bar(x='Classes', y='Samples',  width=1, rot=90, figsize=(15, 15))
  ax.set_title(f'{fold} sample distribution')
  ax.set_ylabel('Number of samples(pixel)')
  plt.show()
