import albumentations as A
import cv2
import os

# Definir las transformaciones
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),  # Volteo horizontal
    A.RandomBrightnessContrast(p=0.2),  # Variaciones de brillo y contraste
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
    A.Blur(blur_limit=3, p=0.1),  # Aplicar desenfoque leve
    A.GaussNoise(p=0.1),  # Añadir ruido gaussiano
    A.CLAHE(p=0.1),  # Ecualización adaptativa
])

# Función para aplicar Data Augmentation y guardar la imagen
def augment_and_save(image_path, output_dir, index):
    # Leer la imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplicar augmentación
    augmented = augmentation(image=image)['image']

    # Convertir de nuevo a BGR para guardar con OpenCV
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

    # Generar un nombre único para el archivo de salida
    original_name = os.path.splitext(os.path.basename(image_path))[0]  # Sin extensión
    new_name = f"{original_name}_aug.jpg"
    output_path = os.path.join(output_dir, new_name)

    # Guardar la imagen augmentada
    cv2.imwrite(output_path, augmented)

# Directorios
input_dir = "datasets/flying_strawberries-classify/train/madura"  # Directorio de entrada
output_dir = "augmentation/madura"  # Directorio de salida

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Aplicar augmentación a todas las imágenes del directorio
for index, image_file in enumerate(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, image_file)
    augment_and_save(image_path, output_dir, index)

print(f"Data Augmentation aplicada y guardada en: {output_dir}")


