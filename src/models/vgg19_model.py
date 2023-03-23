from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.vgg19 import VGG19
from keras.models import Model

class DetectorModelBaseVGG19:
    def __init__(self, weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=["no", "yes"]):
        self.vgg19 = VGG19(weights=weights, include_top=include_top, input_shape=input_shape)
        self.classes = [cl for cl in classes] 
        self.model = None
        self.custom_layers = []
        self.fitted = False
        self.compiled = False
        
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
        assert self.model is not None, "Build model before fitting."
        assert not self.fitted, "Model already fitted."
        assert self.compiled, "Must compile model before fitting."

        history = self.model.fit(
                    train_X,
                    train_Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(test_X, test_Y))

        self.fitted = True

        return history

    def compile(self, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
        assert self.model is not None, "Build model before compiling."
        assert not self.compiled, "Model already compiled."

        self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
                )
        self.compiled = True

    def add(self, layer):
        self.custom_layers.append(layer)

        return self

    def save(self, save_path):
        self.model.save(save_path)

    def score(self, test_X, test_Y, verbose=0):
        return self.model.evaluate(test_X, test_Y, verbose=verbose)
