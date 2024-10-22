import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_processing import process_imgs  # Asegúrate de tener esta función
from model import ViTModel  # Importa tu modelo

device = torch.device("cuda")

# Cargar y procesar los datos
data_directory = os.path.join(os.path.dirname(__file__), '..', 'Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet')
(X_train, X_val, y_train, y_val, label_mapping) = process_imgs(data_directory)

# Asegúrate de que X_train y X_val sean tensores de PyTorch con la forma correcta
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(y_train, dtype=torch.long)  
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2) 
y_val = torch.tensor(y_val, dtype=torch.long)

print(f"Tamaño de conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño de conjunto de prueba: {len(X_val)}")
print(f"Diccionario de etiquetas: {label_mapping}")

# Crear DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Inicializar el modelo
num_classes = len(label_mapping)
model = ViTModel(num_classes).to(device)

# Definir el optimizador y la función de pérdida
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 10

for epoch in range(num_epochs):
    # Establece el modelo en modo entrenamiento
    model.train()  
    total_loss = 0
    total_correct = 0

    for images, labels in train_loader:
        # Mover datos a GPU 
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass y optimización
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()

    # Calcula la precisión en el conjunto de entrenamiento
    train_accuracy = total_correct / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

    model.eval() 
    val_loss = 0
    val_correct = 0

    with torch.no_grad(): 
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Guardamos el modelo
torch.save(model.state_dict(), "el_modelinio.pth")
