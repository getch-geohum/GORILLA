import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def dice_coef(y_true, y_pred, smooth=1e-7):
    
    assert y_true.shape == y_pred.shape, f'Reference and predicted label shapes {y_true.shape} and {y_pred.shape} are not the same!'
    y_pred = torch.softmax(y_pred, dim=1) # expecting the last layer is a logit 
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersect = torch.sum(y_true_f * y_pred_f, axis=-1)
    denom = torch.sum(y_true_f + y_pred_f, axis=-1)
    return torch.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)



def compute_iou(predicted, actual, num_calsses=8, dst=None):
  '''predicted, actual are supposed to be a shape of [b,h, w] or [b, c, h, w] where b,c h, w are batch, height, width and chnnnel(which is 1) '''
  assert predicted.shape == actual.shape, f"Predicted and actual shaps {predicted.shape} and {actual.shape} respectively are not the same "
  iou=np.zeros((num_calsses),dtype=float)
  for k in range(num_calsses):
      a = (predicted == k).int()
      b = (actual == k).int()
      intersection = torch.sum(a*b)
      deno = a + b
      union = torch.sum((deno>0).int())
      iou[k]=intersection/(union + 1e-6)
  print(f'per class ious: {iou.tolist()}')
  if dst == 'digitalglobe':
    mean_iou=(1/(num_calsses-1))*np.sum(iou)
  else:
     mean_iou=(1/num_calsses)*np.sum(iou)
  return mean_iou


def computePixelAccuracy(P, R, average='macro'):
  assert P.shape == R.shape, 'Predicted and reference shapes are not the same'


  if not isinstance(P, np.ndarray):
     P = P.numpy().astype(np.uint8).ravel()
  else:
     P = P.astype(np.uint8).ravel()
  if not isinstance(R, np.ndarray):
     R = R.numpy().astype(np.uint8).ravel()
  else:
     R = R.astype(np.uint8).ravel()

  accuracy  = accuracy_score(R, P)
  precision = precision_score(P, R, average=average, zero_division=0.0)
  recall = recall_score(P, R, average=average, zero_division=0.0)
  f1 = f1_score(P, R, average=average, zero_division=0.0)

  print("accuracy", accuracy)
  print("precision", precision)
  print("recall", recall)
  print("f1 score", f1)





