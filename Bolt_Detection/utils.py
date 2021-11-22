import os
import numpy as np
import torch
from PIL import Image
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import random
from datetime import datetime
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
import timm


def unfreeze(model,percent=0.25):
    l = int(np.ceil(len(model._modules.keys())* percent))
    l = list(model._modules.keys())[-l:]
    print(f"unfreezing these layer {l}",)
    for name in l:
        for params in model._modules[name].parameters():
            params.requires_grad_(True)

def check_freeze(model):
    for name ,layer in model._modules.items():
        s = []
        for l in layer.parameters():
            s.append(l.requires_grad)
        print(name ,all(s))