import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model

from keras.utils import to_categorical

# random_seed = 42
# img_size = 224  # ResNet50 uses 224x224 input size by default
# current_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(current_dir, '../../data/raw/brain_tumor')
# classes = ["no", "yes"]

# X = []
# Y = []
# for cl in classes:
#     folder = os.path.join(data_dir, cl)
#     for img in os.listdir(folder):
#         resized_img = cv2.resize(cv2.imread(
#             os.path.join(folder, img)), (img_size, img_size))
#         X.append(resized_img)
#         Y.append(0 if cl == "no" else 1)

# X = np.array(X)
# Y = np.array(Y)
# Y = to_categorical(Y, num_classes=2)
# train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

class DetectorModelBaseResNet:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self._build_model()

    def _build_model(self):
        resnet50_base = tf.keras.applications.resnet50ResNet50(weights='imagenet', include_top=False,
                                 input_shape=self.input_shape)
        x1 = resnet50_base.output
        x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
        x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
        x1 = tf.keras.layers.Dropout(0.5)(x1)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x1)

        self.model = Model(inputs=[resnet50_base.input], outputs=[predictions])

        for layer in resnet50_base.layers:
            layer.trainable = False

        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_X, train_Y, batch_size=32, epochs=30, validation_split=0.2):
        train_Y = to_categorical(train_Y, num_classes=2)
        history = self.model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_split=validation_split)
        return history

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def evaluate(self, test_X, test_Y):
        test_Y = to_categorical(test_Y, num_classes=2)
        score = self.model.evaluate(test_X, test_Y, verbose=0)
        return score