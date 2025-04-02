# SOURCE: https://colab.research.google.com/drive/1bUFH6F6nP6ole7sQoD6lNk0dAFgZdMAm

import os
os.system("wget https://perso.esiee.fr/~najmanl/DeepLearning/PoolNet.zip")
os.system("unzip PoolNet.zip")
os.system("pip install higra")

import torch
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import numpy as np

import higra as hg

from skimage.transform import resize

import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider, BoundedIntText

try:
    from utils import * # imshow, locate_resource
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

# ==============================================================================

edgedetector = cv2.ximgproc.createStructuredEdgeDetection(get_sed_model_file())
def computeEdgesSED(img):
  src = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  edges = edgedetector.detectEdges(np.float32(src) / 255.0)
  return edges

# ==============================================================================

from os.path import join
from importlib.machinery import SourceFileLoader
networks = SourceFileLoader('networks', join(poolNetDrive+'networks', '__init__.py')).load_module()

from networks.joint_poolnet import build_model, weights_init

poolNetDrive = '/content/PoolNet/'
imgDrive = poolNetDrive + 'Images/'

model = torch.load(poolNetDrive+'final.pth')

net = build_model("resnet")
net = net.cuda()
net.eval()
net.apply(weights_init)
net.load_state_dict(model)

def computeEdgesPoolNet(net, img):
  EPSILON = 1e-8
  scale = [0.5, 1, 1.5, 2]
  im_size = img.shape[:2]
  multi_fuse = np.zeros(im_size, np.float32)
  for k in range(0, len(scale)):
    im_ = cv2.resize(img, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
    im_ = im_.transpose((2, 0, 1))
    im_ = torch.Tensor(im_[np.newaxis, ...])

    with torch.no_grad():
      im_ = Variable(im_)
      im_ = im_.cuda()
      preds = net(im_, mode=0)
      pred_0 = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
      pred_1 = np.squeeze(torch.sigmoid(preds[1][1]).cpu().data.numpy())
      pred_2 = np.squeeze(torch.sigmoid(preds[1][2]).cpu().data.numpy())
      pred_fuse = np.squeeze(torch.sigmoid(preds[0]).cpu().data.numpy())

      pred = (pred_0 + pred_1 + pred_2 + pred_fuse) / 4
      pred = (pred - np.min(pred) + EPSILON) / (np.max(pred) - np.min(pred) + EPSILON)
      pred = cv2.resize(pred, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)
      multi_fuse += pred

  multi_fuse /= len(scale)
  #multi_fuse = (1 - multi_fuse)
  return multi_fuse
