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
checkpoint = torch.load('best.pth.tar',
                        map_location=device)
if torch.cuda.device_count() > 1:
    print("Using multiple GPUs")
    model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state'], strict=False)
model = model.to(device)
model.eval()


#Load test set
test_images = pd.read_csv('test set')

class CXRDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)
    
    def image_loader(self, image_name):
        image = io.imread(image_name)
        image = (((image - image.min()) / (image.max() - image.min()))*255).astype(np.uint8)
        image = np.stack((image, )*3)
        image = np.transpose(image, (1, 2, 0))
        image = self.augmentations(image)
        return image
    
    def __getitem__(self, index):
        y = self.df.at[self.df.index[index], 'MACE_labels']
        x = self.image_loader(self.df.at[self.df.index[index], 'image_paths'])
        y = torch.tensor([y], dtype=torch.float)
        return x, y


#Extract probabilities
other_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4756, 0.4756, 0.4756], std=[0.3011, 0.3011, 0.3011])
                                    ])
datagen_test = CXRDataset(df = test_images.copy(),augmentations = other_transform)
test_loader = DataLoader(dataset=datagen_test,  shuffle=False, batch_size=8, num_workers=4)
all_probabilities = []
all_image_paths = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probabilities.append(probs.flatten())
        
flattened_probs = np.concatenate(all_probabilities, axis=0)
test_images['predicted_probability'] = flattened_probs


#ROC Curve
output = test_images['predicted_probability']
fpr, tpr, thresholds = roc_curve(test_images['MACE_labels'],output) 
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ResNet50 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.savefig("roc_curve_testset.png", dpi=300, bbox_inches='tight')
print("ROC curve for testset saved")


#Final model predictions for test set
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(optimal_threshold)

preds = []
for num in test_images['predicted_probability']:
    if num > optimal_threshold:
        preds.append(1)
    else:
        preds.append(0)

test_images['model_predictions'] = preds


#Additional metrics
y_true = test_images['MACE_labels']
y_pred = test_images['model_predictions']

cm = confusion_matrix(y_true,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots()
disp.plot(ax=ax)
plt.savefig("confusion_matrix_testset.png", dpi=300, bbox_inches='tight')

tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
specificity = tn / (tn + fp)

print(f"Testset Accuracy: {accuracy:.4f}")
print(f"Testset Precision: {precision:.4f}")
print(f"Testset Sensitivity: {sensitivity:.4f}")
print(f"Testset Specificity: {specificity:.4f}")
print(f"Testset F1-score: {f1:.4f}")

#Save testset with model probabilities & predictions
test_images.to_csv('resnet_test_set_evaluated.csv')
