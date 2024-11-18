import torch
import cv2
import numpy as np
from torchvision import transforms
from VitModel import ImprovedViT
import torch.nn.functional as F

class SignLanguagePredictor:
    def __init__(self, model_path, image_size=128, device=None):
        self.image_size = image_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Definir el mapeo de índices a etiquetas
        self.index_to_label = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E" 
        }
        
        # Cargar el modelo
        self.model = self._load_model(model_path)
        
        # Definir la transformación
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def _load_model(self, model_path):
        model = ImprovedViT(
            image_size=self.image_size,
            patch_size=8,
            in_channels=1,
            num_classes=len(self.index_to_label),  # Ajusta esto según tus clases
            embed_dim=256,
            depth=12,
            num_heads=8,
            mlp_ratio=4.,
            dropout=0.1
        )
        
        # Cargar los pesos del modelo
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model


    def preprocess_image(self, image):
        # Convertir a escala de grises si la imagen es RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar Canny
        edges = cv2.Canny(image, 100, 200)
        
        # Resize a 128x128
        resized = cv2.resize(edges, (self.image_size, self.image_size))
        
        return resized

    def predict(self, image):
        # Preprocesar la imagen
        processed_img = self.preprocess_image(image)
        
        # Convertir a tensor y añadir dimensión de batch
        img_tensor = self.transform(processed_img).unsqueeze(0)
        
        # Mover a device y realizar predicción
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Obtener la predicción con mayor probabilidad
            pred_prob, pred_class = torch.max(probabilities, 1)
            
            # Convertir a valores de Python
            predicted_class = pred_class.item()
            probability = pred_prob.item()
            
            # Obtener la etiqueta correspondiente
            predicted_label = self.index_to_label.get(predicted_class, "Unknown")
            
        return predicted_label, probability

# Función de ayuda para visualización
def visualize_prediction(image, prediction, probability):
    # Crear una copia de la imagen
    img_display = image.copy()
    
    # Añadir texto con la predicción
    text = f"{prediction} ({probability:.2%})"
    cv2.putText(img_display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    
    return img_display

# Ejemplo de uso
def main():
    # Inicializar el predictor
    predictor = SignLanguagePredictor(
        model_path = r"C:\Users\joaqu\OneDrive\Escritorio\6tociclo\procesamiento_imagenes\SordoMudos\src\model\vit_model.pth",
        image_size=128
    )
    
    # Cargar y procesar una imagen
    image_path = r"C:\Users\joaqu\OneDrive\Escritorio\6tociclo\procesamiento_imagenes\SordoMudos\images\b_original.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return
    
    # Realizar predicción
    prediction, probability = predictor.predict(image)
    
    # Visualizar resultado
    result_image = visualize_prediction(image, prediction, probability)
    
    # Mostrar imagen
    cv2.imshow("Prediccion", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()