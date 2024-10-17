import cv2
import albumentations as A
import matplotlib.pyplot as plt
import os
import kagglehub

# Descarga la última versión del dataset y define la carpeta donde se almacenarán las imágenes
path = kagglehub.dataset_download("usmanafzaal/strawberry-disease-detection-dataset")
image_folder = os.path.join(path, "imagenes")  # Carpeta donde se guardarán las imágenes

# Crea la carpeta si no existe
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

print("Path to dataset files:", path)

# Define la transformación
transform = A.Compose([
    A.RandomCrop(width=256, height=256),  # Recorta aleatoriamente la imagen
    A.HorizontalFlip(p=0.5),               # Voltea horizontalmente con probabilidad 0.5
    A.Rotate(limit=40, p=0.5),             # Rota la imagen aleatoriamente hasta 40 grados
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),  # Ajusta el color
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normaliza la imagen
])


def augment_image(image_path):
    # Carga la imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte BGR a RGB
    
    # Aplica las transformaciones
    augmented = transform(image=image)
    augmented_image = augmented['image']
    
    return augmented_image

def save_augmented_image(image, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convierte de nuevo a BGR para guardar en OpenCV


# Aplica las transformaciones y guarda varias imágenes aumentadas
num_augmented_images = 10  # Número de imágenes aumentadas a crear
images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]  # Filtra las imágenes en la carpeta

for idx, image_file in enumerate(images):
    image_path = os.path.join(image_folder, image_file)
    
    for i in range(num_augmented_images):
        augmented_image = augment_image(image_path)
        output_path = os.path.join(image_folder, f'augmented_image_{idx}_{i}.jpg')
        save_augmented_image(augmented_image, output_path)

    print(f'Imágenes aumentadas para {image_file} guardadas en {image_folder}')

# Ejemplo: mostrar una de las imágenes aumentadas
plt.imshow(augmented_image.astype('uint8'))
plt.axis('off')
plt.show()
