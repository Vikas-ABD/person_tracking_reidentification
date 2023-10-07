import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomCNNModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            # Convolutional Layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 640, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten the feature maps
            layers.Flatten(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
        ])
        
        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def summary(self):
        self.model.summary()