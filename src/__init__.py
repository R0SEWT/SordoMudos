import os
import torch 
from torchvision import transforms
from PIL import Image
from model import ViTModel

#configuracion de la GPU
device=torch.device("cuda")

def load_model(weights_path, num_classes):
    model=ViTModel(num_classes).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

#preprocesar de la imagen 
def preprocesar_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    image=Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

#Ponemos la prediccion 
def predict(model,image_path):
    image_tensor=preprocesar_image(image_path)
    image_tensor=image_tensor.to(device)
    with torch.no_grad():
        output=model(image_tensor)
        _, predicted_class=torch.max(output.data, 1)
    return predicted_class.item()

#Funcion principal par aprobar el modelo 
def test_model(image_path):
    num_classes=24
    weights_path=os.path.join(os.path.dirname(__file__), '..', 'el_modelinio.pth')
    model=load_model(weights_path, num_classes)
    predicted_class=predict(model, image_path)
    return predicted_class