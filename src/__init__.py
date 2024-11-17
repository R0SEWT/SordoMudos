import os
import torch
from torchvision import transforms
from PIL import Image
from src.models.CNN.model import RusticModel
from src.models.VIT.VitModel import ViTModel
import matplotlib.pyplot as plt

# Configuración de la GPU
device = torch.device("cuda")

def load_model(weights_path, num_classes):
    #model = RusticModel(num_classes).to(device)
    model = ViTModel(num_classes).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# Preprocesar la imagen
def preprocesar_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  
    ])

    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    return image, transform(image).unsqueeze(0)  # Retornar la imagen y el tensor

# Mostrar la imagen
def mostrar_imagen(imagen):
    plt.imshow(imagen, cmap='gray')  # Usar cmap='gray' para imágenes en escala de grises
    plt.axis('off')  # Ocultar los ejes
    plt.show()

# Ponemos la predicción
def predict(model, image_path):
    image, image_tensor = preprocesar_image(image_path)  # Obtener imagen y tensor
    mostrar_imagen(image)  # Mostrar la imagen preprocesada
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output.data, 1)
    return predicted_class.item()

# Función principal para probar el modelo 
def test_model(image_path, weights_path):
    num_classes = 24
    model = load_model(weights_path, num_classes)
    predicted_class = predict(model, image_path)
    return predicted_class
