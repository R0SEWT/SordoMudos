import os
import cv2
import numpy as np

def load_images(data_dir, target_size=(224,224)):
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                # Solo carga archivos de imagen
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img=cv2.imread(img_path)
                    img=cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(label)

    return images, labels
