import cv2
import tensorflow as tf

# Carga la imagen desde la ruta y la devuelve en formato adecuado.
def load_image(image_path, target_size=(512,512)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    image = cv2.resize(image, target_size)
    return image

# Preprocesa la imagen para que esté lista para el modelo.
def preprocess_image(img):
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...] # Añadir dimensión
    return input_tensor