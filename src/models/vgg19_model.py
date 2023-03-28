import numpy as np
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications.vgg19 import VGG19
from keras.models import Model
from sklearn.model_selection import ParameterGrid


class DetectorModelBaseVGG19:
    def __init__(self, weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=["no", "yes"]):
        self.vgg19 = VGG19(weights=weights, include_top=include_top, input_shape=input_shape)
        self.classes = [cl for cl in classes] 
        self.model = None
        self.custom_layers = []
        
    def build(self):
        for layer in self.vgg19.layers:
            layer.trainable = False

        x = self.vgg19.output
        x = GlobalAveragePooling2D()(x)

        if len(self.custom_layers):
            for layer in self.custom_layers:
                x = layer(x)

        # sigmoid is logistic function, suited for binary classification
        predictions = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=self.vgg19.input, outputs=predictions)

    
    def fit(self, train_X, train_Y, test_X, test_Y, batch_size=32, epochs=30, verbose=1):
        history = self.model.fit(
                    train_X,
                    train_Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(test_X, test_Y))

        return history

    def compile(self, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
        self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
                )
        
    def add(self, layer):
        self.custom_layers.append(layer)

        return self

    def save(self, save_path):
        self.model.save(save_path)

    def score(self, test_X, test_Y, verbose=0):
        assert self.model is not None, "Build model before getting score."
        assert self.compiled, "Model needs to be compiled before getting score."
        assert self.fitted, "Model must be fitted before getting score."

        return self.model.evaluate(test_X, test_Y, verbose=verbose)
    def grid_search(self, param_grid, train_X, train_Y, test_X, test_Y, epochs=30, verbose=1):
        best_model = None
        best_score = -np.inf
        best_params = None
        best_history = None

        param_combinations = ParameterGrid(param_grid)

        self.add(Dense(256, activation='relu'))
        self.add(Dropout(0.5))

        for params in param_combinations:
            print(f"Training with parameters: {params}")

            # Reset model
            self.build()
            self.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=["accuracy"])

            history = self.fit(train_X, train_Y, test_X, test_Y, batch_size=params['batch_size'], epochs=epochs, verbose=verbose)

            score = self.score(test_X, test_Y)

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