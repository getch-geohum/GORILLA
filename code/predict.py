# import matplotlib
# matplotlib.use("TkAgg")
import argparse
import logging
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from my_model import UNet
from dataset import LandDataset, DigitalGlobdataset

import matplotlib.pyplot as plt
# import pylab as plt
from glob import glob
from skimage.io import imsave
from torch.utils.data import DataLoader
from losses_metrics import compute_iou, computePixelAccuracy
from utils import plot_esalndcover, plotdigital_glob, plotConfusionmatrix
# # import tkinter
# import matplotlib
# matplotlib.use("TkAgg")

def predict_batch(model,
                imags,
                device,
                out_threshold=0.5):
    model.eval()

    imags = imags.to(device=device)

    with torch.no_grad():
        output = model(imags.float())
        if model.n_classes > 1:
            probs = F.softmax(output, dim=1)
            return torch.argmax(probs, dim=1)
        else:
            probs = torch.sigmoid(output)
            return (probs>=out_threshold).long()
        
def writeBatch(images, refs, preds, index, save_dir):
    assert images.shape[0] == refs.shape[0] == preds.shape[0], 'The number of images in the image, rmask and predictions are not the same'
    im = images[0].squeeze().permute(1,2,0).cpu().numpy()
    rf = refs[0].squeeze().cpu().numpy()
    pre = preds[0].squeeze().cpu().numpy()
    assert im.shape[:2] == rf.shape == pre.shape, f'the image, reference labels and predicted images hapes are not the same {im.shape}, {rf.shape}, {pre.shape} respectively'
    imsave(fname=f'{save_dir}/images/{index}.tif', arr=im, check_contrast=False)
    imsave(fname=f'{save_dir}/labels/{index}.tif', arr=rf, check_contrast=False)
    imsave(fname=f'{save_dir}/preds/{index}.tif', arr=pre, check_contrast=False)


def test(args):
    print("Save the results: ", args.save)
    if args.save:
      folds = ['preds', 'labels', 'images']
      for fold in folds:
        os.makedirs(f'{args.save_dir}/{fold}', exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.dataset != 'digitalglobe':
      print('Loading digital globe test dataset')
      images = glob(f'{args.data_dir}/images/*.tif')
      labels = glob(f'{args.data_dir}/labels/*.tif')
      print(f'Found {len(images)} datsets for testing from easa land cover datasets')
      
      images = [images[i] for i in range(1, len(images), 2)]  # in the validation dataset the start was 0 and here the start is 1 i.e validation and test samples are non-overlapping
      labels = [labels[i] for i in range(1, len(labels), 2)]

      
      test_data = LandDataset(imgs=images, lbls=labels, n_classes=args.n_class, shape=args.shape)
      loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    else:
        print("Loading digital globe test dataset")
        images = sorted(glob(f'{args.data_dir}/test/*_sat.jpg'))
        labels = sorted(glob(f'{args.data_dir}/test/*_mask.png'))

        test_dataset = DigitalGlobdataset(imgs=images, lbls=labels)
        loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        print(f'Test dataset properly loaded with {len(test_dataset)}')
    
    model = UNet(n_channels=args.n_channel, n_classes=args.n_class)
                         

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    model.to(device=device)
    model.load_state_dict(torch.load(args.model, map_location=device)) # weights_only=True

    logging.info("Model loaded !")

    REFS_ = []
    PREDS_ = []
    RAWS_  = []
    for j, batch in tqdm(enumerate(loader)):
      imgs = batch[0].to(device)
      lbls = batch[1].long()
   
      if j == 0:
        print("\n image and mask shape: ", imgs.shape, lbls.shape)
      seg = predict_batch(model=model,
                          imags=imgs,
                          device=device)
      RAWS_.append(imgs)
      REFS_.append(lbls)
      PREDS_.append(seg)
      
      if args.save:
        writeBatch(images=imgs, refs=lbls, preds=seg, index=j, save_dir=args.save_dir)
    # print('Lengths: ', len(RAWS_), len(PREDS_), len(REFS_)) 
    #Model evaluation
    REFS_ = torch.cat(REFS_, dim=0).squeeze().to(device)
    PREDS_ = torch.cat(PREDS_, dim=0)
    RAWS_ = torch.cat(RAWS_, dim=0).permute(0, 2, 3, 1)

    total_IoU = compute_iou(predicted=PREDS_, actual=REFS_, num_calsses=args.n_class, dst=args.dataset)
    print("MIoU for all class:",total_IoU)
    computePixelAccuracy(P=PREDS_.cpu(), R=REFS_.cpu())

    
    if args.plot:
       plotConfusionmatrix(y_true=REFS_.squeeze().cpu().numpy().ravel().astype(np.uint8), y_pred=PREDS_.squeeze().cpu().numpy().ravel().astype(np.uint8), dataset_type=args.dataset)
       for j in range(3):
        inds = random.sample(list(range(RAWS_.shape[0])), 3)
        if args.dataset == 'digitalglobe':
            plotdigital_glob(imgs=[RAWS_[ind].cpu().numpy() for ind in inds], preds=[PREDS_[ind].cpu().numpy() for ind in inds], refs=[REFS_[ind].cpu().numpy() for ind in inds])
        else:
            plot_esalndcover(imgs=[RAWS_[ind].cpu().numpy() for ind in inds], preds=[PREDS_[ind].cpu().numpy() for ind in inds], refs=[REFS_[ind].cpu().numpy() for ind in inds], fname=f'{args.save_dir}/{j}.png')
        

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='/root/.cache/kagglehub/datasets/getachewworkineh/uganda-landcover/versions/4/results/esalandcover/checkpoints/tt_ff_best_weight.pth', metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--data_dir', '-i', default='/root/.cache/kagglehub/datasets/getachewworkineh/uganda-landcover/versions/4/landcover_data_v2/test', help='folder containing test direectory', required=False)
    parser.add_argument('--save_dir', '-o', default='/content/drive/MyDrive/GORILLA-master/GORILLA-master/results/Predictions/esalandcover', help='Path to save output files')
    parser.add_argument('--save', '-k', action='store_true', help="Do not save the output masks", default=False)
    parser.add_argument('--n_class', '-n', type=float, default=8, help="Number of classes in mask or model definition.")
    parser.add_argument('--n_channel', '-c', type=float, default=4, help="Number of input chnannels in the model definition")
    parser.add_argument('--shape', '-s', type=int, default=256, help="spatial dimension of the input images")
    parser.add_argument('--batch_size', '-b', type=int, help="Batch size", default=1)
    parser.add_argument('--dataset', '-w', type=str, help="dataset type to select", default='esalandcover')
    parser.add_argument('--plot', '-p', action='store_true', help="Do not save the output masks", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test(args=args)
    
    