import cv2
import albumentations as A
import matplotlib.pyplot as plt


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

# Ejemplo de uso
image_path = 'ruta/a/tu/imagen.jpg'  # Reemplaza con la ruta de tu imagen
augmented_image = augment_image(image_path)

# Muestra la imagen aumentada
plt.imshow(augmented_image.astype('uint8'))
plt.axis('off')
plt.show()


import os

def save_augmented_image(image, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    cv2.imwrite(output_path, image)

# Guarda la imagen aumentada
output_path = 'ruta/a/tu/carpeta/augmented_image.jpg'  # Reemplaza con la ruta de salida
save_augmented_image(augmented_image, output_path)

num_augmented_images = 10  # Número de imágenes aumentadas a crear
for i in range(num_augmented_images):
    augmented_image = augment_image(image_path)
    save_augmented_image(augmented_image, f'ruta/a/tu/carpeta/augmented_image_{i}.jpg')
