from __init__ import test_model
import os
import cv2
from data_preprocessing import binarize_with_canny
from skimage import io
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image_path = r"C:\Users\joaqu\OneDrive\Escritorio\6tociclo\procesamiento_imagenes\SordoMudos\images\c_original.jpg"
    
    # Cargar la imagen
    image = io.imread(image_path)

    if image is None:
        path = os.path.dirname(__file__)
        full_path = os.path.join(path, image_path)
        print(f"La imagen no ha sido encontrada en {full_path}")
    else:
        # Redimensionar la imagen
        new_img = cv2.resize(image, (128, 128))
        
        # Binarizar la imagen con Canny
        new_img = binarize_with_canny(new_img)
        
        # Guardar la imagen procesada
        save_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'generated', "new_image1.jpg")
        cv2.imwrite(save_path, new_img)
        print(f"Silueta guardada en: {save_path}")

        # Comprobar si los pesos existen
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'mejor_modelo_vit.pth')
        
        if not os.path.exists(weights_path):
            print(f"El archivo de pesos no se encuentra en {weights_path}")
        else:
            # Hacer las predicciones con el modelo
            predictions = test_model(save_path, weights_path)

            # Definir las etiquetas
            labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}
            print(f"La letra predicha es: {labels[predictions]}")
