import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import models.resnet_model
import models.vgg19_model

random_seed = 42
img_size = 224  # ResNet50 uses 224x224 input size by default
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data/raw/brain_tumor')
classes = ["no", "yes"]

X = []
Y = []
for cl in classes:
    folder = os.path.join(data_dir, cl)
    for img in os.listdir(folder):
        resized_img = cv2.resize(cv2.imread(
            os.path.join(folder, img)), (img_size, img_size))
        X.append(resized_img)
        Y.append(0 if cl == "no" else 1)

X = np.array(X)
Y = np.array(Y)
Y = to_categorical(Y, num_classes=2)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=random_seed)