import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model
from keras.utils import to_categorical
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, auc, confusion_matrix


class DetectorModelBaseVGG19:
    def __init__(self, weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=["no", "yes"]):
        self.classes = [cl for cl in classes]
        self._build(weights, include_top, input_shape)

    def _build(self, weights, include_top, input_shape):
        vgg19 = VGG19(
            weights=weights, include_top=include_top, input_shape=input_shape)
        for layer in vgg19.layers:
            layer.trainable = False

        x = vgg19.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)

        predictions = Dense(2, activation='softmax')(x)

        self.model = Model(inputs=vgg19.input, outputs=predictions)

    def fit(self, train_X, train_Y, test_X, test_Y, batch_size=32, epochs=30, verbose=1):
        train_Y = to_categorical(train_Y, num_classes=len(self.classes))
        test_Y = to_categorical(test_Y, num_classes=len(self.classes))
        history = self.model.fit(
            train_X,
            train_Y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(test_X, test_Y))
        self.history = history

        return history

    def compile(self, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def predict(self, X):
        return self.model.predict(X)

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

    def plot_confusion_matrix(self, test_X, test_Y):
        y_pred = self.predict(test_X)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(to_categorical(test_Y), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                    cbar=False, linewidth=0.5)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def load(self, path):
        self.model = load_model(path)

    def save(self, save_path):
        self.model.save(save_path)

    def score(self, test_X, test_Y, verbose=0):
        test_Y = to_categorical(test_Y, num_classes=len(self.classes))
        return self.model.evaluate(test_X, test_Y, verbose=verbose)

    def grid_search(self, param_grid, train_X, train_Y, test_X, test_Y, epochs=30, verbose=1):
        best_model = None
        best_score = -np.inf
        best_params = None
        best_history = None

        param_combinations = ParameterGrid(param_grid)

        for params in param_combinations:
            print(f"Training with parameters: {params}")

            # Reset model
            self.build()
            self.compile(optimizer=params['optimizer'],
                         loss=params['loss'], metrics=["accuracy"])

            history = self.fit(train_X, train_Y, test_X, test_Y,
                               batch_size=params['batch_size'], epochs=params['epochs'], verbose=verbose)

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
