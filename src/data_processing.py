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

def split_data_by_label(imgs, labels, test_size=0.2, random_st=42):
    unique_labels = np.unique(labels)
    imgs_train = []
    imgs_test = []
    labels_train = []
    labels_test = []
    
    for label in unique_labels:
        # Filtra las imágenes y etiquetas para la letra específica
        imgs_of_label = [img for img, lbl in zip(imgs, labels) if lbl == label]
        labels_of_label = [lbl for lbl in labels if lbl == label]
        
        #Makes split for each letter in 0.20 and 0.8
        imgs_train_label, imgs_test_label, labels_train_label, labels_test_label = train_test_split(
            imgs_of_label, labels_of_label, test_size=test_size, random_state=random_st
        )
        #Almacena los datos separados por pruebas
        imgs_train.extend(imgs_train_label)
        imgs_test.extend(imgs_test_label)
        labels_train.extend(labels_train_label)
        labels_test.extend(labels_test_label)
        #View how many files are in test label
        print(f"Letra '{label}': {len(labels_test_label)} elementos prueba")

    return imgs_train, imgs_test, labels_train, labels_test

def process_imgs(data_dir):
    imgs, labels = load_images(data_dir)
    encoded_labels, label_to_index = label_encoder(labels)  
    imgs_train, imgs_test, labels_train, labels_test = split_data_by_label(imgs, encoded_labels)
    return imgs_train, imgs_test, labels_train, labels_test, label_to_index

data_directory = os.path.join(os.path.dirname(__file__), '..', 'Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet')

(X_train, X_val, y_train, y_val, label_mapping) = process_imgs(data_directory)
print ("final")
print(f"Tamaño de conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño de conjunto de prueba: {len(X_val)}")
print(f"Diccionario de etiquetas: {label_mapping}")
