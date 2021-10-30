import os
import copy
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, plot_confusion_matrix


def plot_TL_history(args, history, model_name=None):
    if model_name == None:
        model_name = "CNN"
    num_epochs = args.num_epochs
    filename = os.path.join(args.saved_figures, f"./TL_history_{model_name}.png")

    fig, axes = plt.subplots(2, 1, figsize=(8,8))
    fig.tight_layout(pad=5)

    iters = np.arange(num_epochs) + 1
    fig.suptitle(f'TL History for {model_name}')

    axes[0].set_title('Loss over epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[0].plot(iters, history['train_loss'], label='Training Loss') 
    axes[0].plot(iters, history['val_loss'], label='Validation Loss')
    axes[0].legend(loc='best')

    axes[1].set_title('Accuracy over epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    axes[1].plot(iters, history['train_acc'], label='Training Accuracy') 
    axes[1].plot(iters, history['val_acc'], label='Validation Accuracy')
    axes[1].legend(loc='best')
    
    plt.savefig(filename, dpi=300)
    fig.show()


def plot_ConfMatrix(args, clf, x_test, y_test, filename=None):
    if filename == None:
        filename = "conf_mat"
    filename = os.path.join(args.saved_figures, f"./{filename}.png")
    plot_confusion_matrix(clf, x_test,y_test, values_format='d')
    plt.savefig(filename, dpi=300)
    plt.show()
