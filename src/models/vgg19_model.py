from keras.layers import Dropout, GlobalAveragePooling2D, Dense 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19

vgg19_base = VGG19(weights='imagenet', include_top=False,
                   input_shape=(img_size, img_size, 3))
x1 = vgg19_base.output
x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
x1 = tf.keras.layers.Dropout(0.5)(x1)
predictions = tf.keras.layers.Dense(len(classes), activation='softmax')(x1)

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

class DetectorModelBaseVGG19:
    def __init__(self, weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=["no", "yes"]):
        self.vgg19 = VGG19(weights=weights, include_top=include_top, input_shape=input_shape)
        self.classes = [] 
        
        # create new layers with vgg19 as a base
        x = self.vgg19.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)

        predictions = Dense(len(self.classes), activation='softmax')(x)
    
    def fit(train_X, trian_Y, test_X, test_Y, batch_size=32, epochs=30, verbose=1):
        pass

