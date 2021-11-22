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
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.decorators import auto_move_data
from numbers import Number
from typing import List
from functools import singledispatch
from fastcore.dispatch import typedispatch
from pytorch_lightning.core.decorators import auto_move_data
from ensemble_boxes import ensemble_boxes_wbf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from objdetecteval.metrics.coco_metrics import get_coco_stats
from pytorch_lightning import Trainer


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
        
        
def run_wbf(predictions, image_size=1024, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels