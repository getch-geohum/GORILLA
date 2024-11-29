import torch
import torch.nn.functional as F
from tqdm import tqdm

from losses_metrics import dice_coef


def evaluate_model(model, loader, device, criterion):
  model.eval()
  
  tot = []
  
  for batch in tqdm(loader):
    imgs, masks = batch[0], batch[1]
    imgs = imgs.to(device=device)
    masks = masks.to(device=device)

    with torch.no_grad():
      mask_pred = model(imgs.float())
      vloss = criterion(mask_pred, masks.long()).item()
      tot.append(vloss)
  model.train()
  return sum(tot)/len(tot)
