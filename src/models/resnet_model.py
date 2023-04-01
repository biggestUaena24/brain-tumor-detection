import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from keras.models import Model, load_model
from keras.utils import to_categorical
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, auc, confusion_matrix


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
        x1 = tf.keras.layers.Dropout(0.6)(x1)
        x1 = tf.keras.layers.Dense(512, activation='relu')(x1)
        x1 = tf.keras.layers.Dropout(0.6)(x1)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x1)

        self.model = Model(inputs=[resnet50_base.input], outputs=[predictions])

        for layer in resnet50_base.layers:
            layer.trainable = False

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train_X, train_Y, batch_size=32, epochs=30, validation_split=0.2, verbose=1):
        train_Y = to_categorical(train_Y, num_classes=2)
        history = self.model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                                 validation_split=validation_split)
        self.history = history
        return history

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def compute_roc_curve(self, test_X, test_Y):
        y_true = np.argmax(to_categorical(test_Y, num_classes=2), axis=1)
        y_pred = self.predict(test_X)
        y_pred_pos = y_pred[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_pos)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def plot_roc_curve(self, test_X, test_Y):
        fpr, tpr, roc_auc = self.compute_roc_curve(test_X, test_Y)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'],
                 label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def plot_confusion_matrix(self, test_X, test_Y):
        y_pred = self.predict(test_X)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(to_categorical(test_Y), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='coolwarm', cbar=False, linewidth=0.5)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def score(self, test_X, test_Y):
        test_Y = to_categorical(test_Y, num_classes=2)
        score = self.model.evaluate(test_X, test_Y, verbose=0)
        return score

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def grid_search(self, param_grid, train_X, train_Y, test_X, test_Y, epochs=30, verbose=1):
        best_model = None
        best_score = -np.inf
        best_params = None
        best_history = None

        param_combinations = ParameterGrid(param_grid)

        for params in param_combinations:
            print(f"Training with parameters: {params}")

            self._build_model()
            self.compile(
                optimizer=params['optimizer'], loss=params['loss'], metrics=["accuracy"])
            history = self.fit(
                train_X, train_Y, batch_size=params['batch_size'], epochs=epochs, validation_split=0.2)
            score = self.score(test_X, test_Y)

            if score[1] > best_score:
                best_model = self.model
                best_score = score[1]
                best_params = params
                best_history = history

            print(f"Accuracy: {score[1]}")

        self.model = best_model
        self.history = best_history

        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_score}")

        return best_history, best_params
