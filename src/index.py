from __init__ import test_model
import os
import cv2
from data_preprocessing import binarize_with_canny
from skimage import io
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image_path = r"C:\Users\joaqu\OneDrive\Escritorio\6tociclo\procesamiento_imagenes\SordoMudos\images\a_original.jpg"
    #image = cv2.imread(image_path)
    #image = cv2.imread(r"/home/rosewt/Documentos/codigos2024/SordoMudos/images/a_original.jpg")
    image = io.imread(image_path)


    if image is None:
        path = os.path.dirname(__file__)
        full_path = os.path.join(path, image_path)
        print(f"La imagen no ha sido encontrada en {full_path}")
    else:
        new_img = binarize_with_canny(image)
        save_path = os.path.join(os.path.dirname(__file__), '..', 'images','generated', "new_image1.jpg")
        cv2.imwrite(save_path, new_img)
        print(f"Silueta guardada en: {save_path}")

        weights_path = os.path.join(os.path.dirname(__file__),'modelo_VIT.pth')
        
        if not os.path.exists(weights_path):
            print(f"El archivo de pesos no se encuentra en {weights_path}")
        else:
            predictions = test_model(save_path, weights_path)

            labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
            print(f"La letra predicha es: {labels[predictions]}")
