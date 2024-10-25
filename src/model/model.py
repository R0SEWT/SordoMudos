import tensorflow as tf
from tensorflow.keras.layers import Dropout

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
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),         
        # Aplanar la salida
        tf.keras.layers.Flatten(),  
        Dropout(.30),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_clasess, activation='softmax')
    ])
    # usar GPU
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model



