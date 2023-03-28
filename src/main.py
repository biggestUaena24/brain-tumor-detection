import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from models.resnet_model import DetectorModelBaseResNet
from models.vgg19_model import DetectorModelBaseVGG19

random_seed = 42


def load_data(random_state, test_size=0.2, from_path=None):
    if from_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '../data/processed/brain_tumor')
    else:
        data_dir = from_path
    classes = ["no", "yes"]

    X = []
    Y = []
    for cl in classes:
        folder = os.path.join(data_dir, cl)
        for img in os.listdir(folder):
            img_data = cv2.imread(os.path.join(folder, img))
            X.append(img_data)
            Y.append(0 if cl == "no" else 1)

    X = np.array(X)
    Y = np.array(Y)

    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


train_X, test_X, train_Y, test_Y = load_data(random_seed)

param_grid_resnet = {
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
vgg_history, vgg_best_params = vgg_model.grid_search(
    param_grid_vgg, train_X, train_Y, test_X, test_Y, epochs=10)

resnet_model = DetectorModelBaseResNet()
resnet_history, resnet_best_params = resnet_model.grid_search(
    param_grid_resnet, train_X, train_Y, test_X, test_Y, epochs=10)
