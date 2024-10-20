import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo errores
from modules.img_preprocessing import load_image, preprocess_image
from modules.visualization import draw_detections, display_image
from modules.detection import load_efficientdet_model, detect_fruit

# Cargar las imágenes
img_carpeta = "img"
img_archivos = os.listdir(img_carpeta)

# Cargar el modelo de detección
modelo_deteccion = load_efficientdet_model()

for img_archivo in img_archivos:
    img_ruta = os.path.join(img_carpeta, img_archivo)

    # Cargar y preprocesar la imagen
    img = load_image(img_ruta)
    img_preprocesada = preprocess_image(img)

    # Detectar las frutillas
    detecciones = detect_fruit(img_preprocesada, modelo_deteccion)
    print(f"Número de detecciones: {len(detecciones)}")

    # Visualizar los resultados
    img_con_detecciones = draw_detections(img, detecciones)
    display_image(img_con_detecciones)