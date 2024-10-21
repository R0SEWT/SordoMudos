import os
import cv2
import numpy as np

def data_augmented(image, target_size=(224, 224)):
    augmented_imgs = []

    # Aplicamos la rotación aleatoria
    angle = np.random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    rotated_img = cv2.resize(rotated_img, target_size)  # Redimensionar
    augmented_imgs.append(rotated_img)

    # Escalamiento de las imágenes
    scale = np.random.uniform(0.8, 1.2)
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
    scaled_image = cv2.resize(scaled_image, target_size)  # Redimensionar
    augmented_imgs.append(scaled_image)

    # Ajuste de brillo aleatorio
    bright = np.random.randint(-50, 50)
    bright_img = cv2.convertScaleAbs(image, alpha=1, beta=bright)
    bright_img = cv2.resize(bright_img, target_size)  # Redimensionar
    augmented_imgs.append(bright_img)

    return augmented_imgs

def load_images(data_dir, target_size=(224, 224)):
    images = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)

        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                # Solo carga archivos de imagen
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, target_size)

                    # Agregamos imágenes originales
                    images.append(img)
                    labels.append(label)

                    # Generamos 3 imágenes aumentadas
                    augmented_imgs = data_augmented(img, target_size)
                    for aug_img in augmented_imgs:
                        images.append(aug_img)
                        labels.append(label)

    return images, labels
