import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from models.resnet_model import DetectorModelBaseResNet
from models.vgg19_model import DetectorModelBaseVGG19

random_seed = 42
img_size = 224  # ResNet50 uses 224x224 input size by default
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..\data\\raw\\brain_tumor')
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
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

param_grid_Resnet = {
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['categorical_crossentropy'],
    'batch_size': [32, 64]
}

param_grid_vgg = {
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['binary_crossentropy'],
    'batch_size': [32, 64]
}

vgg_model = DetectorModelBaseVGG19()
vgg_history, vgg_best_params = model.grid_search(param_grid_vgg, train_X, train_Y, test_X, test_Y, epochs=10)

resnet_model = DetectorModelBaseResNet()
resnet_history, resnet_best_params = model.grid_search(param_grid, train_X, train_Y, test_X, test_Y, epochs=10)

