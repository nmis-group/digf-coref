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