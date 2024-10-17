import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(data_dir):
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                # Solo carga archivos de imagen
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    images.append(img_path)
                    labels.append(label)

    return images, labels

def label_encoder(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_index[label] for label in labels])  
    return encoded_labels, label_to_index

def split_data(imgs, labels, test_size=0.2, random_st=42):
    return train_test_split(imgs, labels, test_size=test_size, random_state=random_st)


def process_imgs(data_dir):
    imgs, labels = load_images(data_dir)
    encoded_labels, label_to_index = label_encoder(labels)  
    imgs_train, imgs_test, labels_train, labels_test = split_data(imgs, encoded_labels)

    #view
    print ("test load")
    print(f"Total de imágenes cargadas: {len(imgs)}")
    print(f"Primeras 5 etiquetas: {labels[:5]}")
    print ("encoding")
    print(f"Etiquetas codificadas: {encoded_labels[:5]}")
    print(f"Diccionario de mapeo: {label_to_index}")
    print ("split")
    print(f"Tamaño de conjunto de entrenamiento: {len(imgs_train)}")
    print(f"Tamaño de conjunto de prueba: {len(imgs_test)}")


    return imgs_train, imgs_test, labels_train, labels_test, label_to_index

data_directory = os.path.join(os.path.dirname(__file__), '..', 'data_set')

(X_train, X_val, y_train, y_val, label_mapping) = process_imgs(data_directory)
print ("final")
print(f"Tamaño de conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño de conjunto de prueba: {len(X_val)}")
print(f"Diccionario de etiquetas: {label_mapping}")