import numpy as np
import pandas as pd
import os
from os import path
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from skimage import io
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder,StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import sys
import time
import torchvision.utils as vutils
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import ast
import re
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torchxrayvision as xrv


#Load model
model = xrv.models.ResNet(weights="resnet50-res512-all")
last_layer = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(256, 1),
    nn.Sigmoid()
)
model.model.fc = last_layer
model.op_threshs = None


#Load the fine-tuned weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('/mnt/storage/MACE_extraction/src/image_classification/resnet_train/resnet_train-04/best.pth.tar',
                        map_location=device)
if torch.cuda.device_count() > 1:
    print("Using multiple GPUs")
    model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state'], strict=False)
model = model.to(device)
model.eval()
