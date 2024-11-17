import os
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from PIL import Image


import s_preprocessing as s_preprocessing
from data_augmentation import load_images
from s_preprocessing import save_silhouette




def label_encoder(labels):
    """
    Codifica etiquetas de texto en valores numéricos.
    """
    try:
        unique_labels = np.unique(labels)
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        encoded_labels = np.array([label_to_index[label] for label in labels])  
        logging.info("Codificación de etiquetas completada con éxito.")
        return encoded_labels, label_to_index
    except Exception as e:
        logging.error(f"Error al codificar etiquetas: {e}")
        raise



def split_data_by_label(imgs, labels, test_size=0.2, val_size=0.2, random_st=42, DEBUG=False):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba, manteniendo la proporción de etiquetas.
    """
    try:
        unique_labels = np.unique(labels)
        imgs_train, imgs_val, imgs_test = [], [], []
        labels_train, labels_val, labels_test = [], [], []
        
        for label in unique_labels:
            imgs_of_label = [img for img, lbl in zip(imgs, labels) if lbl == label]
            labels_of_label = [lbl for lbl in labels if lbl == label]
            
            # Primera división para separar el conjunto de prueba
            imgs_temp, imgs_test_label, labels_temp, labels_test_label = train_test_split(
                imgs_of_label, labels_of_label, test_size=test_size, random_state=random_st
            )
            
            # Segunda división para separar entrenamiento y validación
            imgs_train_label, imgs_val_label, labels_train_label, labels_val_label = train_test_split(
                imgs_temp, labels_temp, test_size=val_size, random_state=random_st
            )
            
            imgs_train.extend(imgs_train_label)
            imgs_val.extend(imgs_val_label)
            imgs_test.extend(imgs_test_label)
            labels_train.extend(labels_train_label)
            labels_val.extend(labels_val_label)
            labels_test.extend(labels_test_label)
            
            if DEBUG:
                logging.info(f"Etiqueta '{label}': Train={len(labels_train_label)}, Val={len(labels_val_label)}, Test={len(labels_test_label)}")
        
        if not DEBUG:
            logging.info(f"Conjunto de entrenamiento: {len(labels_train)} elementos")
            logging.info(f"Conjunto de validación: {len(labels_val)} elementos")
            logging.info(f"Conjunto de prueba: {len(labels_test)} elementos")
            logging.info("División de datos simétrica por etiqueta completada con éxito")

        return imgs_train, imgs_test, labels_train, labels_test, imgs_val, labels_val
    except Exception as e:
        logging.error(f"Error al dividir datos por etiqueta: {e}")
        raise


def save_split_data(imgs_train, imgs_test, labels_train, labels_test, imgs_val, labels_val, output_dir='split_data'):
    try:
        splits = {
            'train': (imgs_train, labels_train),
            'test': (imgs_test, labels_test),
            'val': (imgs_val, labels_val)
        }
        
        for split_name, (images, labels) in splits.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            for idx, (img, label) in enumerate(zip(images, labels)):
                img_path = os.path.join(split_dir, f'{split_name}_{label}_{idx}.jpg')  # Mantener .jpg
                # aqui se guarddaban con ipy
                image = Image.fromarray((img * 255).astype(np.uint8))  # Ajustar según la escala de tus datos
                image.save(img_path)  # Guarda como .jpg
            
        logging.info(f"Datos guardados exitosamente en {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error al guardar los datos divididos: {e}")
        raise

def preprocess(data_dir, method='canny', output_dir='augmented_images'):
    """
    Carga y procesa imágenes con aumento de datos y guarda siluetas.
    """
    try:
        # imprimir el directorio acutal
        if not os.path.exists(data_dir + '../{}'.format(output_dir)):
            load_images(data_dir, output_dir = output_dir)
        logging.info("Imágenes cargadas y aumentadas con éxito.")
        silhouettes = save_silhouette(method=method, output_dir=output_dir, input_dir=data_dir)
        logging.info("Siluetas guardadas correctamente.")
        return silhouettes
    except Exception as e:
        logging.error(f"Error en el preprocesamiento de imágenes: {e}")
        raise


def process_imgs(data_dir, method='canny', data_output = 'split_data'):
    """
    Procesa imágenes, codifica etiquetas y las divide en conjuntos de entrenamiento, validacion y test.
    """
    try:
        augmented_dir = data_output + '/../augmented/'
        imgs, labels = preprocess(data_dir, method, output_dir=augmented_dir)  
        encoded_labels, label_to_index = label_encoder(labels)  
        imgs_train, imgs_test, labels_train, labels_test, imgs_val, labels_val = split_data_by_label(imgs, encoded_labels)
        
        save_split_data(imgs_train, imgs_test, labels_train, labels_test, imgs_val, labels_val, output_dir=data_output)


        imgs_train = np.array(imgs_train)
        imgs_test = np.array(imgs_test)
        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)
        
        logging.info("Proceso de imágenes completado.")
        return imgs_train, imgs_test, labels_train, labels_test, imgs_val, labels_val, label_to_index
    except Exception as e:
        logging.error(f"Error al procesar imágenes: {e}")
        raise




