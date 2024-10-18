import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import kagglehub

# Descargar la versión más reciente del dataset
path = kagglehub.dataset_download("trainingdatapro/ripe-strawberries-detection")

print("Ruta a los archivos del dataset:", path)

# Ruta a la carpeta que contiene las imágenes del dataset
image_folder = os.path.join(path, 'images')  

print("Archivos en el directorio de imágenes:")
image_files = os.listdir(image_folder)
print(image_files)

# Crear una carpeta para guardar las imágenes transformadas
output_folder = os.path.join(os.getcwd(), 'output_images')
os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe

# Transformaciones
transform = A.Compose([
    A.RandomRotate90(p=1),  # Rota la imagen en ángulos aleatorios de 90 grados
    A.HorizontalFlip(p=1),  # Voltea la imagen horizontalmente
    A.VerticalFlip(p=0.5),  # Voltea la imagen verticalmente con 50% de probabilidad
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  # Cambios suaves en brillo y contraste
    A.GaussNoise(var_limit=(8.0, 20.0), p=1),  # Agregar ruido gaussiano suave
    A.ElasticTransform(alpha=2.0, sigma=50, p=1),  # Deformación elástica
    A.CLAHE(clip_limit=3.0, p=1),  # Aumento de contraste con CLAHE
])

# Iterar sobre cada archivo
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        continue  # Saltar a la siguiente imagen si hay un error

    # Convertir de BGR (formato OpenCV) a RGB para mostrar correctamente con matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplicar la transformación
    transformed = transform(image=image_rgb)
    transformed_image = transformed['image']

    # Convertir de RGB a BGR para guardarla con OpenCV
    transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

    # Ruta para guardar la imagen transformada
    output_path = os.path.join(output_folder, f'transformed_{image_name}')  # Guardar con prefijo 'transformed_'
    cv2.imwrite(output_path, transformed_image_bgr)

    print(f"Imagen transformada guardada en: {output_path}")

print("Transformación de todas las imágenes completada.")
