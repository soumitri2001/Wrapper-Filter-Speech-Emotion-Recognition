import os
import copy
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_score, classification_report, plot_confusion_matrix, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN

import warnings
warnings.filterwarnings('ignore')

from utils.dataset import get_dataloader
from utils.visualize import plot_TL_history, plot_ConfMatrix
from model import SER_Network, train_model, extract_features
from utils.feature_selection import *
from PasiLuukka import pasi_luukka
from WOA_FS import whale_optim_algo

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### CLI arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.getcwd(), help='path to directory where image data is stored')
parser.add_argument('--max_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate for training')
parser.add_argument('--optimizer', type=str, default="SGD", help='optimizer for training: SGD / Adam')
parser.add_argument('--saved_models', type=str, default=os.getcwd()+"/saved_models/", help='path to directory to saved models')
parser.add_argument('--saved_figures', type=str, default=os.getcwd()+"/figures/", help='path to directory to save figures')
parser.add_argument('--saved_features', type=str, default=os.getcwd()+"/saved_features/", help='path to directory to save features')
args = parser.parse_args()


### create the storage directories if not present ###
if not os.path.isdir(args.saved_models):
        os.mkdir(args.saved_models)
if not os.path.isdir(args.saved_figures):
        os.mkdir(args.saved_figures)
if not os.path.isdir(args.saved_features):
        os.mkdir(args.saved_features)


### getting dataloaders ###
classes_to_idx, data_loader = get_dataloader(args)
print('-------------------------------------------------------------------------')

NUM_CLASSES = len(classes_to_idx)
PHASES = data_loader.keys()
for phase in PHASES:
    print(f'Length of {phase} loader = {len(data_loader[phase])}')
print('-------------------------------------------------------------------------')

### initializing model, optimizer, loss function
model = SER_Network(num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

if args.optimizer.upper() not in ['SGD', 'ADAM']:
    args.optimizer = 'SGD'

if args.optimizer.upper() == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)


### train CNN model on dataset ###
print("MODEL TRAINING BEGINS: ")
print('-------------------------------------------------------------------------')
model, history = train_model(args, model, criterion, optimizer, data_loader, PHASES) 
print('-------------------------------------------------------------------------')

### saving final model ###
torch.save(model.state_dict(), os.path.join(args.saved_models, f'{model.MODEL_NAME}.pth'))

### get history to CPU from cuda ###
for i in range(num_epochs):
    history['train_acc'][i] = history['train_acc'][i].cpu().numpy().item() 
    history['val_acc'][i] = history['val_acc'][i].cpu().numpy().item()

### plot learning curves ###
plot_TL_history(args, history, model.MODEL_NAME)

### feature extraction using the trained CNN ###
print("FEATURE EXTRACTION BEGINS: ")
print('-------------------------------------------------------------------------')

args.batch_size = 1 # for feature extraction
_, data_loader = get_dataloader(args)

features, true_labels, paths = [], [], []
for phase in data_loader.keys():
    features, true_labels, paths = extract_features(features, true_labels, paths, model, data_loader[phase], phase)
    print(f'Total feature set after {phase} phase extraction: {len(features)}')
print('-------------------------------------------------------------------------')

# get feature vectors from torch tensors
features, true_labels, paths = get_features(features, true_labels, paths)

# save features to csv file
features_df = pd.DataFrame(features)
features_df['filename'] = paths.copy()
features_df['label'] = true_labels.copy()
features_df.to_csv(os.path.join(args.saved_features, f"{model.MODEL_NAME}_features.csv"),index=False)
print(f'feature set obtained by {model.MODEL_NAME} saved successfully !')
print('-------------------------------------------------------------------------')


### Feature selection using Passi-Luukka's method of Fuzzy Entropy and Similarity Measures ###
X = features_df.iloc[:,0:(features_df.shape[1]-2)].copy()
y = features_df['label'].copy()

X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)

sol_filter = pasi_luukka(X, y)
X1 = sol_filter.ranked_features[:, 0:256].copy()


### Feature selection using Whale Optimization Algorithm ###
data = Data()
data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(X1, y, test_size=0.2, shuffle=False, random_state=1)

sol_WOA = whale_optim_algo(num_agents=40, max_iter=100, data=data)
print('-------------------------------------------------------------------------')


### Final classification using KNN classifier on selected feature subset ###
validate_FS(X1, y, sol_WOA.best_agent, save=False)
print('-------------------------------------------------------------------------')
