# # Preprocesamiento de Datos
# Este notebook se encarga de cargar y preprocesar los datos necesarios para el entrenamiento del modelo.

# Importación de bibliotecas
import cv2
import skimage
import matplotlib.pyplot as plt
import numpy as np
import os

import logging
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

# # ruta de este archivo
# this_file_path = os.path.abspath(os.path.dirname(__file__))

# log_path = os.path.join(this_file_path, "../../logs/data_preprocessing.log")

# # Configuración del logging
# logging.basicConfig(
#     filename="./logs/data_preprocessing.log",
#     filemode="a",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )


# ## Funciones de preprocesamiento

# ## Carga de Datos
# Cargamos los datos desde su fuente y verificamos los primeros registros para entender su estructura.
# 



def convert_to_silhouette(img, threshold=127, binarization_threshold=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, silhouette = cv2.threshold(gray, binarization_threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    

    silhouette = cv2.threshold(silhouette, 127, 255, cv2.THRESH_BINARY)[1] 

    if np.mean(silhouette) > threshold:
        silhouette = cv2.bitwise_not(silhouette)
    
    return silhouette


def binarize_image_with_canny(img, weak_th=None, strong_th=None):
    if img is None:
        logging.error("La imagen no puede ser None.")
        raise ValueError("La imagen no puede ser None")
    
    # Conversión de la imagen a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Reducción de ruido con un filtro gaussiano
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Cálculo de los gradientes
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    
    # Conversión de coordenadas cartesianas a polares
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Establecimiento de umbrales mínimo y máximo
    mag_max = np.max(mag)
    if weak_th is None:
        weak_th = mag_max * 0.1
    if strong_th is None:
        strong_th = mag_max * 0.5
    
    # Aplicación del algoritmo de Canny
    _, binary = cv2.threshold(mag, weak_th, strong_th, cv2.THRESH_BINARY)
    
    return binary



def binarize_with_canny_cv2(img, weak_th=None, strong_th=None):
    if img is None:
        logging.error("La imagen no puede ser None.")
        raise ValueError("La imagen no puede ser None")
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        pass
    
    # Aplicación directa del algorimo de Canny de OpenCV
    if weak_th is None:
        weak_th = 100
    if strong_th is None:
        strong_th = 200
    binary = cv2.Canny(img, weak_th, strong_th)
    
    return binary

def is_valid_image(file_name):
    return file_name.lower().endswith(('.png', '.jpg', '.jpeg'))

def get_new_image_name(file_name, suffix="_s"):
    name, ext = os.path.splitext(file_name)
    return f"{name}{suffix}{ext}"

def process_and_save_silhouette(img_path, label, output_dir, method="canny"):
    img = cv2.imread(img_path)

    # Seleccionar el método de binarización
    method = method.lower()

    if method == "canny":
        silhouette = binarize_image_with_canny(img)
    elif method == "canny_cv2":
        silhouette = binarize_with_canny_cv2(img)
    elif method == "silhouette":
        silhouette = convert_to_silhouette(img)

    new_name = get_new_image_name(os.path.basename(img_path))

    # Ruta de salida para guardar la imagen
    save_dir = os.path.join(output_dir, label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, new_name)

    # Guardar la imagen procesada
    cv2.imwrite(save_path, silhouette)
    return silhouette


def save_silhouette(input_dir='augmented_images', output_dir='images_processed', method="canny"):
    images = []
    labels = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.info(f"Procesando imágenes en {input_dir} y guardando en {output_dir}.")
    # Procesar cada imagen en el directorio de entrada
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if is_valid_image(file):
                img_path = os.path.join(root, file)
                
                # Extraer la etiqueta de la carpeta
                label = os.path.basename(os.path.dirname(img_path))
                silhouette = process_and_save_silhouette(img_path, label, output_dir, method="canny" )

                # Agregar la imagen y la etiqueta a las listas
                images.append(silhouette)
                labels.append(label)

    logging.info(f'Procesamiento de imágenes completado. {len(images)} imágenes procesadas con {method}.')    
    return images, labels

