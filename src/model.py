import torch
import torch.nn as nn
import torch.nn.functional as F

# Definimos el modelo
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        # Capas convolucionales
        self.red_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Aplanado de las dimensiones
        self.flatten = nn.Flatten()
        
        # Capa fully connected
        self.fc1 = nn.Linear(256 * 13 * 13, 256)  # Ajustar el tamaño según sea necesario
        self.dropout = nn.Dropout(0.30)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Reescalado (normalización de los valores de la imagen)
        x = x / 255.0
        
        # Pasar por capas convolucionales
        x = self.red_conv(x)

        # Aplanado
        x = self.flatten(x)
        
        # Capa fully connected con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Capa de salida
        x = self.fc2(x)
        return x

