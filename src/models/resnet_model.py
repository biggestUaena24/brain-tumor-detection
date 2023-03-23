import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model

from keras.utils import to_categorical

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

resnet50_base = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False,
                         input_shape=(img_size, img_size, 3))
x1 = resnet50_base.output
x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
x1 = tf.keras.layers.Dropout(0.5)(x1)
predictions = tf.keras.layers.Dense(len(classes), activation='softmax')(x1)

model = Model(inputs=[resnet50_base.input], outputs=[predictions])

for layer in resnet50_base.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 32
epochs = 30

history = model.fit(
    train_X,
    train_Y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(test_X, test_Y)
)
model.summary()

score = model.evaluate(test_X, test_Y, verbose=0)
print("loss:", score[0])
print("accuracy:", score[1])