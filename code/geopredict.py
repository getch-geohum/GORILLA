import argparse
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
import os
import torch
from tqdm import tqdm
from glob import glob
from skimage import transform
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import from_levels_and_colors


from my_model import UNet

def predict(model, image, device, tile=True):
    if tile:
        out = torch.empty(size=(512,512))
        for i in range(2):
            for j in range(2):
                sub = image[:,:, i*256:i*256+256, j*256:j*256+256]
                sub = sub.to(device)
                out_ = model(sub.float())
                out_ = torch.nn.Softmax(dim=1)(out_)
                out_ = torch.argmax(out_, dim=1).squeeze().long().cpu() # .numpy()
                out[i*256:i*256+256, j*256:j*256+256] = out_
    else:
        image = image.to(device)
        out = model(image.float())  # assuming both are at the same device
        out = torch.nn.Softmax(dim=1)(out)
        out = torch.argmax(out, dim=1).squeeze().long().cpu() # .numpy() # this gives image without batch dim as i9niger data type
    return out.numpy()+1


def preprocess(img, resze=False, size=512, channel_first=True):
    if resze:
        assert size is not None, 'Resie dimension is required.'
    
    if channel_first:
        size = (4, size, size)
    else:
        size = (size, size, 4)
    
    # print('Image shape before: ', img.shape)
    img = np.resize(img, new_shape=size)
    img = (img-img.min())/((img.max()-img.min())+ 1e-7)

    # print('Image shape: ', img.shape)
    
    img = torch.from_numpy(img.astype(float)) 
    
    if not channel_first:
        img = img.permute(2, 0, 1)
        
    return img.unsqueeze(0)




def predict_moaic(files, n_channel, n_class, checkpoint, out_dir, shape):
    '''
    files: list of file pathes for rasters
    n_channel: number of channels in the image
    n_class: number of coutput classes from the model
    chkpoint_dir: checkpoint or model weight full path
    out_dir: the directory to save predicted and mosaiced image
    '''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=n_channel, n_classes=n_class, bilinear=True)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    # model = load_model(channel=n_channel, n_class=n_class, weights=checkpoint)
    

    mosaic_container = []
    for file in tqdm(files):
        ins = rasterio.open(file)
        profile = ins.profile
        profile.update(count=1, dtype=np.uint8)
        arr = ins.read() # assuming its 4 chnnel image
        c, h, w = arr.shape

        pr_array = preprocess(img = arr, resze=False, size = shape, channel_first=True)
        pr_array = pr_array.to(device)
        with torch.no_grad():
            pr_array = predict(model=model, image=pr_array, device=device, tile=True)

        pr_array_resize = transform.resize(image=pr_array,output_shape=(h, w)) # np.resize(pr_array, new_shape = (h, w))

        memfile = MemoryFile() # as memfile:
        with memfile.open(**profile) as dataset:
          dataset.write(pr_array_resize, 1)

            # with memfile.open() as o_dataset:
        mosaic_container.append(memfile.open())
    
    # print(mosaic_container)
    mosaic, out_trans = merge(mosaic_container)
    profile.update({"height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "count": 1,
                     "dtype": np.uint8
                       }
                        )

    out_fp = f'{out_dir}/mosaic.tif'

    with rasterio.open(out_fp, "w", **profile) as dest:
        dest.write(mosaic.astype(np.uint8))

def visualize(file=None):
    arr = imread(file)
    values = list(range(0, 10))
    colors = ["#E8E8E8", "#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#f0f0f0", "#0064c8", "#0096a0", "#FFFFFF"]
    cmap, norm = from_levels_and_colors(values, colors, 'max')
    classes = ['background', "Tree cover", "Shrubland", 'Grassland','Cropland', 'Built-up', 'Bare/sparse vegetation', 'Permanent water bodies', 'Herbaceous wetland', 'n']

    fig, ax = plt.subplots(figsize=(18, 14))
    cax = ax.imshow(arr, cmap=cmap, norm=norm)

    cbar = fig.colorbar(cax, ticks=[i+0.5 for i in list(range(0, 10))], orientation='vertical', extendrect = False, extendfrac='auto')
    cbar.ax.set_yticklabels(classes)  # horizontal colorbar
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--checkpoint', dest='checkpoint', type=str, default='/root/.cache/kagglehub/datasets/getachewworkineh/uganda-landcover/versions/4/results/esalandcover/checkpoints/tt_ff_best_weight.pth', help='Load model from a .pth file')
    parser.add_argument('-n', '--n_class', default=8, type=int, help='Number of classes in the mask/label and or model')
    parser.add_argument('-c', '--n_channel', default=4, type=int, help='Number of channels in the image')
    parser.add_argument('-t', '--data_dir', type=str, default='/root/.cache/kagglehub/datasets/getachewworkineh/uganda-landcover/versions/4/landcover_data_v2/test/images', help='train data folder')
    parser.add_argument('-d', '--save_dir', type=str, default='/content/drive/MyDrive/GORILLA-master/GORILLA-master/results/Predictions/esalandcover', help='directory to save the checkpoint and results')
    parser.add_argument('-x', '--ext', type=str, help='image and labnel file extension', default='.tif')
    parser.add_argument('-s', '--shape', type=int, default=512, help='image shape')
    parser.add_argument('-v', '--visualize', action='store_true', help='Whether to plot the mosaic fill scene for inspection')
    return parser.parse_args()



def main(args):
    
    files = glob(f'{args.data_dir}/*{args.ext}')
    # print('total number of test files: ', len(files))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    predict_moaic(files=files,
                  n_channel=args.n_channel,
                  n_class=args.n_class,
                  checkpoint=args.checkpoint,
                  out_dir=args.save_dir,
                  shape=args.shape)
    
    if args.visualize:
        file = args.save_dir + '/mosaic.tif'
        visualize(file=file)

    

if __name__ == '__main__':
    args = get_args()
    main(args=args)
