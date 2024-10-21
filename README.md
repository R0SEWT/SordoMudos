# SordoMudos Traductor de Lenguaje de Señas con Vision Transformer

este proyecto utiliza un modelo de Vision Transformer (ViT) para traducir el lenguaje de señas peruano a texto. El dataset consiste en imágenes estaticas de señas para cada letra del alfabeto.

## Estructura del Proyecto
- `Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet/`: Submodulo que contiene las imágenes de entrenamiento, validación y prueba
- `src/`: Codigo fuente del proyecto

## Instalación
1. Clona el repositorio y el submodulo:
```bash
git submodule init
git submodule update
```
1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
    ```
## Bibliogrgafia del data set
https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet
