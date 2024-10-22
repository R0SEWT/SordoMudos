import torch
import torch.nn as nn
import torch.nn.functional as F

# Definimos el modelo
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Aplanado de las dimensiones
        self.flatten = nn.Flatten()
        
        # Capa fully connected
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Ajustar el tamaño según sea necesario
        self.dropout = nn.Dropout(0.30)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Reescalado (normalización de los valores de la imagen)
        x = x / 255.0
        
        # Pasar por capas convolucionales
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Aplanado
        x = self.flatten(x)
        
        # Capa fully connected con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Capa de salida
        x = self.fc2(x)
        return x

