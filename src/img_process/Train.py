import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from VitModel import ImprovedViT
from data_load import process_imgs
# Procesar las imágenes y etiquetas
imgs_train, imgs_test, labels_train, labels_test, label_to_index = process_imgs()

# Convertir las imágenes a tensores de PyTorch y normalizarlas
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5501], std=[0.1045])
])

# Si las imágenes no están en formato tensor, convertirlas aquí
imgs_train_tensor = torch.stack([transform(img) for img in imgs_train])
imgs_test_tensor = torch.stack([transform(img) for img in imgs_test])
imgs_train_tensor = imgs_train_tensor.float()
imgs_test_tensor = imgs_test_tensor.float()

# Convertir las etiquetas a tensores de tipo long
labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

# Crear datasets de entrenamiento y prueba
train_dataset = TensorDataset(imgs_train_tensor, labels_train_tensor)
test_dataset = TensorDataset(imgs_test_tensor, labels_test_tensor)

# Crear DataLoader para la iteración por lotes
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Crear el modelo
model = ImprovedViT(
    image_size=128,          # Tamaño de la imagen
    patch_size=8,            # Tamaño de los parches
    in_channels=1,           # Imagen en escala de grises
    num_classes=len(label_to_index),  # Número de clases
    embed_dim=256,           # Dimensión del embedding
    depth=12,                # Profundidad de bloques del transformador
    num_heads=8,             # Número de cabezas de atención
    mlp_ratio=4.,            # Relación de dimensiones del MLP
    dropout=0.1              # Dropout
)

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir la función de pérdida (CrossEntropyLoss para clasificación)
criterion = nn.CrossEntropyLoss()

# Definir el optimizador (Adam)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
# Función para entrenar el modelo
def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterar por los datos de entrenamiento
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()

            # Optimizar el modelo
            optimizer.zero_grad()

            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Calcular la pérdida y precisión por época
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        
        # Mostrar resultados al final de cada época
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy*100:.2f}%")
    
    print("Entrenamiento completado")


def evaluate_model(model, test_loader, criterion):
    model.eval()  # Cambiar a modo de evaluación
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No calcular gradientes durante la evaluación
        for imgs, labels in tqdm(test_loader, desc="Evaluando..."):
            imgs, labels = imgs.to(device), labels.to(device).long()  # Asegurar que las etiquetas sean de tipo long

            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    print(f"Loss de prueba: {test_loss:.4f}, Precisión de prueba: {test_accuracy*100:.2f}%")


# Entrenar el modelo
train_model(model, train_loader, optimizer, criterion, num_epochs=30)
evaluate_model(model, test_loader, criterion)
torch.save(model.state_dict(), 'vit_model.pth')
