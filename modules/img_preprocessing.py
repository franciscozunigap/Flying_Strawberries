import cv2
import tensorflow as tf
import numpy as np

# Carga la imagen desde la ruta y la devuelve en formato adecuado.
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    return image

# Preprocesa la imagen para que esté lista para el modelo.
def preprocess_image(img, target_size=(512, 512)):
    img = cv2.resize(img, target_size)  # Redimensiona la imagen
    img = img.astype(np.float32) / 255.0  # Normaliza a [0, 1]
    return img