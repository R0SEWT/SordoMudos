import os
from data_processing import process_imgs
from model import create_vit

data_directory = os.path.join(os.path.dirname(__file__), '..', 'Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet')
(X_train, X_val, y_train, y_val, label_mapping) = process_imgs(data_directory)

print(f"Tamaño de conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño de conjunto de prueba: {len(X_val)}")
print(f"Diccionario de etiquetas: {label_mapping}")
# Entrenamiento{}
num_clasess = len(label_mapping)
model = create_vit(num_clasess)

try:
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save("el_modelinio.keras")   
except Exception as e:
    print("Error durante el entrenamiento:", str(e))
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
