import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps
  
   
def ssim_metric(target, prediction):
    tar = target.cpu().numpy()
    pred = prediction.cpu().numpy()
    tar = (tar*255).astype(np.uint8).transpose((1, 2, 0))
    pred = (pred*255).astype(np.uint8).transpose((1, 2, 0))
    ssim_score = ssim(tar, pred, channel_axis=2)
    return torch.tensor(ssim_score)


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps
