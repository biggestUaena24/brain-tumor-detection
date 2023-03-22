import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19

random_seed = 42
img_size = 244
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data/raw/brain_tumor')
classes = ["no", "yes"]

X = []
Y = []
for cl in classes:
    folder = os.path.join(data_dir, cl)
    for img in os.listdir(folder):
        resized_img = cv2.resize(cv2.imread(os.path.join(folder, img)), (img_size, img_size))
        X.append(resized_img)
        Y.append(0 if cl == "no" else 1)

X = np.array(X)
Y = np.array(Y)
Y = to_categorical(Y, num_classes=2)
train_X, train_Y, test_X, test_Y = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

base_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
print(base_vgg.output)