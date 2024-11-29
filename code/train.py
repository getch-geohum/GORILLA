import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from glob import glob
import pandas as pd

from evaluate import evaluate_model
from my_model import UNet

from torch.utils.tensorboard import SummaryWriter
from dataset import LandDataset, DigitalGlobdataset
from torch.utils.data import DataLoader  # , random_split  # Note on random split
from utils import get_class_weight, get_class_weightDG

# torch.autograd.set_detect_anomaly(True)


def train(model,
          device,
          epochs=5,
          batch_size=1,
          lr=0.001,
          save_cp=True,
          n_class=11,
          train_data_dir='',
          val_data_dir='',
          ext='.tif',
          shape=(512, 512),
          save_dir='',
          early_stop=10,
          downscale=False,
          encode=False,
          datasettype=None):

    dir_checkpoint = f'{save_dir}/{datasettype}/checkpoints'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, exist_ok=True)


    if datasettype != 'digitalglobe':
      print('Loading dataset from ESA land cover dataset')
        
      train_impages = sorted(glob(f'{train_data_dir}/train/images/*.tif'))
      train_labels = sorted(glob(f'{train_data_dir}/train/labels/*.tif'))
      train_dataset = LandDataset(imgs=train_impages, lbls=train_labels, n_classes=n_class, shape=shape, one_encoding=encode)
       
      valid_impages = sorted(glob(f'{val_data_dir}/valid/images/*.tif'))
      valid_impages = [valid_impages[i] for i in list(range(0, len(valid_impages),2))]
      valid_labels = sorted(glob(f'{val_data_dir}/valid/labels/*.tif'))
      valid_labels = [valid_labels[i] for i in list(range(0, len(valid_labels),2))]
      valid_dataset = LandDataset(imgs=valid_impages, lbls=valid_labels, n_classes=n_class, shape=shape, one_encoding=encode, subset=True)
      print((len(train_dataset)), len(valid_dataset))
    else:
       print('Loading dataset from Digital globe dataset')
       train_impages = sorted(glob(f'{train_data_dir}/train/*_sat.jpg'))
       train_labels = sorted(glob(f'{train_data_dir}/train/*_mask.png'))
       train_dataset = DigitalGlobdataset(imgs=train_impages, lbls=train_labels)

       valid_impages = sorted(glob(f'{train_data_dir}/valid/*_sat.jpg'))
       valid_labels = sorted(glob(f'{train_data_dir}/valid/*_mask.png'))
       valid_dataset = DigitalGlobdataset(imgs=valid_impages, lbls=valid_labels)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    n_train = len(train_dataset)
    n_valid = len(valid_dataset)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_valid}
        Checkpoint dir:     {save_cp}
        Device:          {device.type}
        Dataset: {datasettype}
    ''')


    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5*len(train_loader), eta_min=1e-8, last_epoch=-1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

    # temp = train_labels+ valid_labels
    if datasettype == 'digitalglobe':
       weights_classes = get_class_weightDG(train_labels+valid_labels, n_class=n_class)
    else:
       weights_classes = get_class_weight(train_labels+valid_labels, n_class=n_class)

    print("Class weight: ", weights_classes.tolist())

    weights_classes = torch.from_numpy(weights_classes)
    weights_classes = weights_classes.to(device=device)

    if n_class > 1:
        criterion = nn.CrossEntropyLoss(weight = weights_classes.float()).to(device=device)
        print('cross entropy loss is selected!')
    else:
        criterion = nn.BCEWithLogitsLoss().to(device=device)
        print('Binary cross entropy loss is selected!')
    
    total_loss = []
    valid_loss = []
    counter = 0   # to monitor the training performance

    tileindex = 0

    for j in range(1, epochs+1):
      model.train()

      batch_loss = []
      for i, batch in enumerate(train_loader):
        imgs = batch[0]
        masks = batch[1]
        if j == 1 and i == 0:
          print("image shape", imgs.shape, masks.shape)
        
        assert imgs.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

        imgs = imgs.to(device=device) 
        masks = masks.to(device)
        # masks = torch.argmax(masks, dim=1).long().to(device=device)
                 
        masks_pred = model(imgs.float())

        loss = criterion(masks_pred, masks.long())
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.01)
        optimizer.step()
        batch_loss.append(loss.item())
        logging.info(f'Epoch: {j}, step: {i}, step_loss: {batch_loss[-1]}, classes: {np.unique(masks.cpu().numpy()).tolist()}')
        # scheduler.step()
      
      b_loss = sum(batch_loss)/len(batch_loss)
      total_loss.append(b_loss)

      val_score = evaluate_model(model=model, loader=val_loader, device=device, criterion=criterion)
      valid_loss.append(val_score)
      logging.info(f'Epoch: {j}, epoch_loss: {total_loss[-1]}: val_loss: {valid_loss[-1]}')

      if datasettype != 'digitalglobe':
        tileindex+=1
        if tileindex>3:
          tileindex = 0

        train_loader.dataset.set_tileindex(tileindex)
        val_loader.dataset.set_tileindex(tileindex)
        


      if j == 1:
        pass
      elif valid_loss[-2]> valid_loss[-1]:
        torch.save(model.state_dict(), dir_checkpoint + '/tt_ff_best_weight.pth')
        counter = 0
      else:
        counter+=1
      
      if counter>=early_stop:
        
        torch.save(model.state_dict(), dir_checkpoint + '/tt_ff_final_weight.pth')
        loss_dict = {'train_loss': total_loss, 'valid_loss': valid_loss}
        df = pd.DataFrame.from_dict(loss_dict)
        df.to_csv(dir_checkpoint + '/tt_ff_train_val_loss.csv')
        break

    torch.save(model.state_dict(), dir_checkpoint + '/tt_ff_final_weight.pth')
    loss_dict = {'train_loss': total_loss, 'valid_loss': valid_loss}
    df = pd.DataFrame.from_dict(loss_dict)
    df.to_csv(dir_checkpoint + '/tt_ff_train_val_loss.csv')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=4e-5, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-n', '--n_class', default=8, type=int, help='Number of classes in the mask/label and or model')
    parser.add_argument('-z', '--onehot', action="store_true", help="one hot encoding of the classes")
    parser.add_argument('-q', '--resize', action="store_true", help="resize the larger tile to a desired image spatial dimension")  
    parser.add_argument('-c', '--n_channel', default=4, type=int, help='Number of channels in the image')
    parser.add_argument('-t', '--train_data_dir', type=str, help='train data folder')
    parser.add_argument('-v', '--val_data_dir', type=str, help='validation data folder')
    parser.add_argument('-x', '--ext', type=str, help='image and labnel file extension', default='.tif')
    parser.add_argument('-s', '--shape', type=int, default=512, help='image shape')
    parser.add_argument('-d', '--save_dir', type=str, help='directory to save the checkpoint and results')
    parser.add_argument('-w','--dataset', type=str, default='esalandcover')

    
    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'Using device {device}')
    
    model = UNet(n_channels=args.n_channel, n_classes=args.n_class, bilinear=True)
    
    logging.info(f'Network:\n'
                 f'\t{args.n_channel} input channels\n'
                 f'\t{args.n_class} output classes \n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    try:
        train(model=model,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              save_cp=True,
              n_class=args.n_class,
              train_data_dir=args.train_data_dir,
              val_data_dir=args.val_data_dir,
              ext=args.ext,
              shape=args.shape,
              save_dir=args.save_dir,
              early_stop=10,
              encode=args.onehot,
              downscale=args.resize,
              datasettype=args.dataset)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    args = get_args()
    main(args=args)
    