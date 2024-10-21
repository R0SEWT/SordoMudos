import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_processing import process_imgs

data_directory = os.path.join(os.path.dirname(__file__), '..', 'Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet')
(X_train, X_val, y_train, y_val, label_mapping) = process_imgs(data_directory)

input_shape = (224, 224, 3)

def create_vit(num_clasess):
    model = tf.keras.Sequential([
        #Tamaño de imagen reescalado
        tf.keras.layers.Input(shape=input_shape),
        #normalizamos las cosas osea las imagenes :p
        tf.keras.layers.Rescaling(1./255),
        
        # Capas convolucionales
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
        
        # Aplanar la salida
        tf.keras.layers.Flatten(),  
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_clasess, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Entrenamiento
num_clasess = len(label_mapping)
model = create_vit(num_clasess)

# Ajustamos el modelo con el conjunto de entrenamiento
try:
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
except Exception as e:
    print("Error durante el entrenamiento:", str(e))
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

