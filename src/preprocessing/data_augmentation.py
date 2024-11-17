#!/usr/bin/env python
# coding: utf-8

# # Aumento de Datos en Imágenes
# Este notebook realiza aumentos de datos en imágenes, incluyendo rotación y escalado, y guarda las imágenes aumentadas en una estructura de carpetas.
# 
# ## Importación de Bibliotecas
# Primero, importamos las bibliotecas necesarias para realizar los aumentos de datos.
# 

# Importación de bibliotecas
import os
import cv2
import numpy as np
import logging


# ## Función: Guardar Imágenes
# Esta función guarda una imagen redimensionada en un directorio especificado, estructurado por etiquetas.


def save_image(image, label, img_name, output_dir, target_size=(224, 224), print=False):
    # Crear directorio basado en la etiqueta
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Redimensionar y guardar la imagen
    img_path = os.path.join(label_dir, img_name)
    resized_img = cv2.resize(image, target_size)
    cv2.imwrite(img_path, resized_img)
    if print:
        logging.info(f"Imagen guardada en: {img_path}")


# ## Función: Aumento de Datos
# La función `data_augmented` realiza aumentos de datos en una imagen, aplicando rotación y escalado. Las imágenes aumentadas se guardan usando `save_image`.
# 

def data_augmented(image, label, img_base_name, output_dir, target_size=(224, 224)):
    augmented_imgs = []

    def augmented_rotate(img, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h))
        rotated_img = cv2.resize(rotated_img, target_size)
        augmented_imgs.append(rotated_img)
        save_image(rotated_img, label, img_base_name + f"_r{angle}.jpg", output_dir)
        logging.info(f"Imagen rotada {angle} grados y guardada.")

    def augmented_escale(img, scale):
        scaled_image = cv2.resize(img, None, fx=scale, fy=scale)
        scaled_image = cv2.resize(scaled_image, target_size)
        augmented_imgs.append(scaled_image)
        save_image(scaled_image, label, img_base_name + f"_s{scale}.jpg", output_dir)
        logging.info(f"Imagen escalada con factor {scale} y guardada.")

    def augmented_bright(img, bright):
        bright_img = cv2.convertScaleAbs(image, alpha=1, beta=bright)
        bright_img = cv2.resize(bright_img, target_size)  
        augmented_imgs.append(bright_img)
        save_image(bright_img, label, img_base_name + "_b.jpg", output_dir)
    
    def augmented_noise(img):
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8) # gaussian_noise
        noisy_image = cv2.add(image, noise)
        noisy_image = cv2.resize(noisy_image, target_size)  
        augmented_imgs.append(noisy_image)
        save_image(noisy_image, label, img_base_name + "_n.jpg", output_dir)


    # Aplicamos la rotación aleatoria
    angle = np.random.uniform(-30, 30)
    #augmented_rotate(image, angle)

    # Escalamiento de las imágenes
    scale = np.random.uniform(0.8, 1.2)
    #augmented_escale(image, scale)
    
    # Ajuste de brillo aleatorio
    bright = np.random.randint(-50, 50)
    #augmented_bright(image, bright)
    return augmented_imgs




def load_images(data_dir, output_dir='augmented_images', target_size=(224, 224)):
    images = []
    labels = []

    # Crea el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)

        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                # Solo carga archivos de imagen
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, target_size)

                    # Agregamos la imagen original a la memoria
                    images.append(img)
                    labels.append(label)

                    # Guardamos la imagen original en el directorio generado
                    img_base_name = os.path.splitext(img_file)[0]
                    save_image(img, label, img_base_name + "_original.jpg", output_dir)

                    # Generamos imágenes aumentadas y las guardamos
                    augmented_imgs = data_augmented(img, label, img_base_name, output_dir)

                    images.extend(augmented_imgs)
                    labels.extend([label] * len(augmented_imgs))

    return images, labels

