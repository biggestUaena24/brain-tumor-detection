import numpy as np
import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import ParameterGrid

from keras.utils import to_categorical


class DetectorModelBaseResNet:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self._build_model()

    def _build_model(self):
        resnet50_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
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

    def grid_search(self, param_grid, train_X, train_Y, test_X, test_Y, epochs=30, verbose=1):
        best_model = None
        best_score = -np.inf
        best_params = None
        best_history = None

        param_combinations = ParameterGrid(param_grid)

        for params in param_combinations:
            print(f"Training with parameters: {params}")

            self._build_model()
            self.model.compile(
                optimizer=params['optimizer'], loss=params['loss'], metrics=["accuracy"])
            history = self.train(
                train_X, train_Y, batch_size=params['batch_size'], epochs=epochs, validation_split=0.2)
            score = self.evaluate(test_X, test_Y)

            if score[1] > best_score:
                best_model = self.model
                best_score = score[1]
                best_params = params
                best_history = history

            print(f"Accuracy: {score[1]}")

        self.model = best_model

        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_score}")

        return best_history, best_params
