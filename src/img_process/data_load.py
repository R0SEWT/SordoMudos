import os
import numpy as np
from sklearn.model_selection import train_test_split
from img_preprocess import img_canny
from collections import defaultdict

def label_encoder(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_index[label] for label in labels])  
    return encoded_labels, label_to_index

def split_data_by_label(imgs, labels, test_size=0.3, random_st=42):
    # Agrupar imágenes y etiquetas por cada etiqueta única en un solo paso
    label_dict = defaultdict(list)
    for img, label in zip(imgs, labels):
        label_dict[label].append(img)
    
    imgs_train, imgs_test = [], []
    labels_train, labels_test = [], []

    for label, img_list in label_dict.items():
        # Dividir entre entrenamiento y prueba por cada etiqueta
        imgs_train_label, imgs_test_label = train_test_split(
            img_list, test_size=test_size, random_state=random_st
        )
        
        # Agregar las imágenes y etiquetas a los conjuntos de entrenamiento y prueba
        imgs_train.extend(imgs_train_label)
        imgs_test.extend(imgs_test_label)
        labels_train.extend([label] * len(imgs_train_label))
        labels_test.extend([label] * len(imgs_test_label))
        
        print(f"Letra '{label}': {len(imgs_test_label)} elementos prueba")
    
    return np.array(imgs_train), np.array(imgs_test), np.array(labels_train), np.array(labels_test)

def print_label_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nDistribución de etiquetas:")
    for label, count in zip(unique_labels, counts):
        print(f"Etiqueta '{label}': {count} imagenes")


def process_imgs():
    # Procesa las imágenes y las etiquetas
    imgs, labels = img_canny(os.path.join(os.path.dirname(__file__), '..', '..', 'captures'))  

    # Codificar las etiquetas como números
    encoded_labels, label_to_index = label_encoder(labels)  
    print_label_distribution(encoded_labels)
    # Dividir las imágenes en conjuntos de entrenamiento y prueba
    imgs_train, imgs_test, labels_train, labels_test = split_data_by_label(imgs, encoded_labels)
    
    return imgs_train, imgs_test, labels_train, labels_test, label_to_index