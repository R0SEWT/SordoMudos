
import torch
import os
import cv2
from skimage import io
import numpy as np

# Función para cargar y preprocesar la imagen
def preprocess_image(image_path):
    image = io.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"La imagen no ha sido encontrada en {image_path}")
    
    # Redimensionar la imagen a 128x128 y binarizarla con Canny
    new_img = cv2.resize(image, (128, 128))
    new_img = binarize_with_canny(new_img)
    
    # Normalizar la imagen (opcional) y convertirla en un tensor
    new_img = new_img.astype(np.float32) / 255.0  # Normalizar la imagen
    new_img = np.expand_dims(new_img, axis=0)  # Añadir dimensión de canal
    new_img = np.expand_dims(new_img, axis=0)  # Añadir dimensión de batch
    new_img_tensor = torch.tensor(new_img)
    
    return new_img_tensor

# Función para cargar el modelo y hacer predicciones
def test_model(image_tensor, weights_path):
    # Definir el modelo con la misma configuración que el entrenamiento
    model = ImprovedViT(
        image_size=128,
        patch_size=8,
        in_channels=1,  # Porque binarizamos la imagen en escala de grises
        num_classes=5,  # Ajustar según el número de clases
        embed_dim=256,
        depth=12,
        num_heads=8,
        dropout=0.1
    )
    
    # Cargar los pesos
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()  # Poner el modelo en modo evaluación
    
    # Hacer la predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

if __name__ == '__main__':
    image_path = r"C:\Users\joaqu\OneDrive\Escritorio\6tociclo\procesamiento_imagenes\SordoMudos\images\c_original.jpg"
    
    try:
        # Preprocesar la imagen
        image_tensor = preprocess_image(image_path)
        
        # Comprobar si los pesos existen
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'mejor_modelo_vit.pth')
        
        if not os.path.exists(weights_path):
            print(f"El archivo de pesos no se encuentra en {weights_path}")
        else:
            # Hacer la predicción con el modelo
            prediction = test_model(image_tensor, weights_path)
            
            # Definir las etiquetas
            labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}
            print(f"La letra predicha es: {labels[prediction]}")
    
    except FileNotFoundError as e:
        print(e)
