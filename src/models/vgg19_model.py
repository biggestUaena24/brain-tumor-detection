import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

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
        resized_img = cv2.resize(cv2.imread(
            os.path.join(folder, img)), (img_size, img_size))
        X.append(resized_img)
        Y.append(0 if cl == "no" else 1)

X = np.array(X)
Y = np.array(Y)
Y = to_categorical(Y, num_classes=2)
train_X, train_Y, test_X, test_Y = train_test_split(
    X, Y, test_size=0.2, random_state=random_seed)

vgg19_base = VGG19(weights='imagenet', include_top=False,
                   input_shape=(img_size, img_size, 3))
x1 = vgg19_base.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
predictions = Dense(len(classes), activation='softmax')(x1)

model = Model(inputs=[vgg19_base.input], outputs=[predictions])

for layer in vgg19_base.layers:
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

# model.save("/some/path")
