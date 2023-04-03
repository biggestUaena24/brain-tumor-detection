from sklearn.model_selection import train_test_split
from models.resnet_model import DetectorModelBaseResNet
from models.vgg19_model import DetectorModelBaseVGG19
from data.make_dataset import load_data

random_seed = 42


# limit 600 means that there are going to be 600 images that don't have brain tumor
# and 600 images that have brain tumour
# 600 is a sweet point to have for a balance model (no overfitting, or underfitting)
X, Y = load_data(target="train", limit=600)
train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.2, random_state=random_seed)

param_grid_resnet = {
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['categorical_crossentropy'],
    'batch_size': [32, 64],
    'epochs': [10, 20, 30],
}

param_grid_vgg = {
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['binary_crossentropy'],
    'batch_size': [32, 64],
    'epochs': [10, 20, 30],
}

"""
WARNING: KEEP IN MIND THAT ITERATING 10, 20, and 30 EPOCHS
         FOR GRID SEARCH WILL TAKE A HUGE AMOUNT OF TIME.
         CONSIDER USING GPU ACCELERATION WITH CUDA.
         OUR CODE USES TENSORFLOW UNDERNEATH IT,
         SO IT SHOULD BE READY TO USE GPU WHEN IT IS AVAILABLE.
"""

resnet_model = DetectorModelBaseResNet()
resnet_history, resnet_best_params = resnet_model.grid_search(
    param_grid_resnet, train_X, train_Y, test_X, test_Y, verbose=1)
resnet_model.plot_accuracy()
resnet_model.plot_roc_curve(test_X, test_Y)
resnet_model.plot_confusion_matrix(test_X, test_Y)

vgg_model = DetectorModelBaseVGG19()
vgg_history, vgg_best_params = vgg_model.grid_search(
    param_grid_vgg, train_X, train_Y, test_X, test_Y, verbose=1)
vgg_model.plot_accuracy()
vgg_model.plot_roc_curve(test_X, test_Y)
vgg_model.plot_confusion_matrix(test_X, test_Y)


# Saving trained model, uncomment lines below
# resnet_model.save(path)
# vgg_model.save(path)
