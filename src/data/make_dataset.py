import os
import numpy as np
import tensorflow as tf

from pathlib import Path


def preprocess_image(img_path, target_size=(224, 224)):
    assert type(target_size) == tuple, "Target size must be of type tuple"
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, target_size)
    img /= 255.0  # normalize
    return img


def load_data(target="train", limit=200, randomize=True):
    # for now the method only loads all the data as yes or no
    current_file_path = Path(__file__).resolve()

    X = []
    Y = []
    # load images that don't have brain tumour
    if target == "train":
        data_dir = os.path.join(current_file_path, '../../data/raw/Training')
    else:
        data_dir = os.path.join(current_file_path, '../../data/raw/Testing')

    no_dir = os.path.join(data_dir, "no")

    files = os.listdir(no_dir)
    if randomize:
        np.random.shuffle(files)

    count = 0
    for img_name in files:
        if count == limit:
            break
        img = preprocess_image(os.path.join(no_dir, img_name))
        X.append(img)
        Y.append(0)
        count += 1

    # load images that have brain tumour
    labels = ['pituitary', 'meningioma', 'glioma']
    n = len(labels)
    limit_per_label = [limit // n if i != n -
                       1 else limit // n + limit % n for i in range(n)]
    for i in range(n):
        limit = limit_per_label[i]
        label = labels[i]
        count = 0
        folder = os.path.join(data_dir, label)
        imgs = os.listdir(folder)
        if randomize:
            np.random.shuffle(imgs)
        for img_name in imgs:
            if count == limit:
                break
            img = preprocess_image(os.path.join(folder, img_name))
            X.append(img)
            Y.append(1)
            count += 1

    X = np.array(X)
    Y = np.array(Y)

    return X, Y
