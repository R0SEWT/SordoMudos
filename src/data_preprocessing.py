import cv2
import numpy as np
import os

def to_silhouette(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #sse le aplica el umbral para segmentar la mano
    _, silhouette = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(silhouette) > 127:
        silhouette = cv2.bitwise_not(silhouette)
    return silhouette

def save_silhouette(input_dir=os.path.join(os.path.dirname(__file__), '..', 'augmented_images'), output_dir=os.path.join(os.path.dirname(__file__), '..', 'images_processed')):
    images = []
    labels = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Recorrer todas las imágenes del directorio de entrada
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                silhouette = to_silhouette(img)

                # Extraer la etiqueta desde el nombre de la carpeta
                label = os.path.basename(os.path.dirname(img_path))
                labels.append(label)

                # Obtener el nombre original del archivo y agregar "_s" antes de la extensión
                name, ext = os.path.splitext(file)
                new_name = f"{name}_s{ext}"
                # Crear la estructura de carpetas en la salida
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # Guardar la imagen de silueta en la carpeta correspondiente
                save_path = os.path.join(output_path, new_name)
                cv2.imwrite(save_path, silhouette)
                
                # Agregar la imagen a la lista
                images.append(silhouette)
    
    return images, labels