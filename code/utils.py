import numpy as np
from tqdm import tqdm
from skimage.io import imread
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def onehot(x, nc):
    h, w = x.shape
    empty = np.zeros((h, w, nc), x.dtype)
    for i in range(0, nc):
        c_i = x==i+1
        empty[:,:,i] = c_i.astype(x.dtype)
    return empty

def toFloat(x):
    return x.float()


def train_testsplit(files,part,train_ration=0.7, report=True):
    N = len(files)
    tn = int(N*train_ration)
    vt_n = N-tn

    gap = int(N/vt_n)
    v_inds = list(range(0, N, gap))

    vt_files = [files[i] for i in v_inds]
    train_files = list(set(files).symmetric_difference(set(vt_files)))
    valid_files = [vt_files[i] for i in range(0, len(vt_files), 2)]
    test_files = [vt_files[i] for i in range(1, len(vt_files), 2)]
    file_dict = {'train':train_files, 'valid':valid_files, 'test':test_files}

    if report:
       print(part, ':', len(file_dict[part]))
    return file_dict[part]

def remapLC(lc_image):
    empty = np.copy(lc_image)
    vals = list(range(10, 101, 10))
    vals.insert(-1, 95)
    maps = list(range(1, 12, 1))
    # print(list(zip(vals, maps)))
    for i in range(len(vals)):
        empty[empty==vals[i]] = maps[i]
    return empty


def remapLC_filter(lc_image):
    empty = np.copy(lc_image)
    vals = [10, 20, 30, 40, 50, 60, 80, 90] 
    maps = list(range(1, 9, 1))
  
    for i in range(len(vals)):
        empty[empty==vals[i]] = maps[i]
    return empty

def remamDGlob(lc_image, shape):
  mapping = [np.array([0,255,255]),
              np.array([255,255,0]),
              np.array([255,0,255]),
              np.array([0,255,0]),
              np.array([0,0,255]),
              np.array([255,255,255]),
              np.array([0,0,0])
              ]  # this is based on provided dataset encoding
  lc_class = np.zeros((shape, shape, len(mapping)), int)
  for j, inds in enumerate(mapping):
    val = lc_image == inds
    val = np.sum(val.astype(int), axis=-1) == 3
    lc_class[:,:, j] = val.astype(int)
  return lc_class


def get_class_weight(files, n_class):
  print('Computing class weight..!')
  class_freq = np.zeros(n_class, dtype=float)
  
  for j in tqdm(range(len(files))):
    img = imread(files[j])
    img = remapLC_filter(img)-1

    for i in range(n_class):
       out = img == i
       class_freq[i]+=np.sum(out.astype(np.uint8))

    class_weight = 1 - class_freq/class_freq.sum()
    return class_weight
  

def get_class_weightDG(files, n_class):
  print('Computing class weight..!')
  class_freq = np.zeros(n_class, dtype=float)
  
  for j in tqdm(range(len(files))):
    img = imread(files[j])
    for i in range(n_class):
       out = img == i
       class_freq[i]+=np.sum(out.astype(np.uint8))

    class_weight = 1 - class_freq/class_freq.sum()

    return class_weight
  


def plot_esalndcover(imgs, preds, refs):
  
  classes = ["Tree cover", "Shrubland", 'Grassland','Cropland', 'Built-up', 'Bare/sparse vegetation',  'Permanent water bodies', 'Herbaceous wetland']
  colors = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#0064c8", "#99ff99"]  #  "#f0f0f0", "#0096a0", "#00cf75", "#fae6a0"
  values = list(range(0, len(classes)))
  cmap, norm = from_levels_and_colors(values, colors, 'max')

  row = 3
  col = 3
  size = (18, 18)
  titles = ['input', 'prected', 'reference']
  fig, axs = plt.subplots(nrows=row, ncols=col, figsize=size)
  alls = []
  for item in list(zip(imgs, preds, refs)):
     alls.extend(list(item))
  
  print('length of items: ', len(alls))

  subs = []
  for i, (ax, data) in enumerate(zip(axs.flat, alls)):
     if i%3 !=0:
      subs.append(ax.imshow(data, norm=norm, cmap=cmap))
     else:
      subs.append(ax.imshow(np.dstack([data[:,:, 2-i] for i in range(3)]), norm=norm))
  cbar = fig.colorbar(subs[2], ax=axs, ticks=list(range(0, len(colors))), orientation='vertical', extendrect = False)
  cbar.ax.set_yticklabels(classes)

  for j in range(3):
     axs[0,j].set_title(titles[j])  
  plt.show()



def plotdigital_glob(imgs, preds, refs):
  
  classes = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
  colors = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#f0f0f0"]  # "#0064c8", "#0096a0", "#00cf75", "#fae6a0"
  values = list(range(0, len(classes)))
  cmap, norm = from_levels_and_colors(values, colors, 'max')

  row = 3
  col = 3
  size = (18, 18)

  titles = ['input', 'prected', 'reference']
  fig, axs = plt.subplots(nrows=row, ncols=col, figsize=size)
  alls = []
  for item in list(zip(imgs, preds, refs)):
     alls.extend(list(item))
  
  print('length of items: ', len(alls))

  subs = []
  for i, (ax, data) in enumerate(zip(axs.flat, alls)):
     if i%3 !=0:
      subs.append(ax.imshow(data, norm=norm, cmap=cmap))
     else:
      subs.append(ax.imshow(np.dstack([data[:,:, 2-i] for i in range(3)]), norm=norm))
  cbar = fig.colorbar(subs[2], ax=axs, ticks=list(range(0, len(colors))), orientation='vertical', extendrect = False)
  cbar.ax.set_yticklabels(classes)

  for j in range(3):
     axs[0,j].set_title(titles[j]) 
  plt.show()



def plotConfusionmatrix(y_true, y_pred, dataset_type):
   if dataset_type == 'digitalglobe':
      classes = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
   else:
      classes = ["Tree cover", "Shrubland", 'Grassland','Cropland', 'Built-up', 'Bare/sparse vegetation',  'Permanent water bodies', 'Herbaceous wetland']
   cm = confusion_matrix(y_true=y_true.ravel(), y_pred=y_pred.ravel(), normalize='true', labels= list(range(0, len(classes))))
   cm = np.round(cm, 4)

   cmp = ConfusionMatrixDisplay(cm, display_labels=classes)
   cmp.plot()
   cmp.ax_.set(xlabel='Predicted class', ylabel='True class')
   plt.title(f"Confusion matrix normalized with Number of True classes")

   # plt.imshow(cm)
   # plt.xticks(ticks=[a for a in list(range(0, len(classes)))], labels=classes, rotation=90)
   # plt.yticks(ticks=[a for a in list(range(0, len(classes)))], labels=classes, rotation=45)
   # plt.show()

