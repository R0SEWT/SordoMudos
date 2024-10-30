from __init__ import test_model
import os
import cv2
from data_preprocessing import to_silhouette

if __name__ == '__main__':
    image_path = "../images/a_valeria_3.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"la iamgen no ha sido encontrada en {image_path}")
    else:
        new_img = to_silhouette(image)
        save_path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'images'), "new_image.jpg")
        cv2.imwrite(save_path, new_img)
        print(f"Silueta guardada en: {save_path}")
        predictions = test_model(save_path)

        labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 
                  9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 
                  17: 's', 18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'}

        print(f"La letra predicha es: {labels[predictions]}")
