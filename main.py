import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo errores
from config import instalar_dependencias
from modules.img_preprocessing import load_image, preprocess_image
from modules.visualization import draw_detections, display_image
from modules.detection import load_efficientdet_model, detect_fruit

ruta_requerimientos = "requirements.txt"
# instalar_dependencias(ruta_requerimientos)

# Cargar las imágenes
img_carpeta = "img"
img_archivos = os.listdir(img_carpeta)

# Cargar el modelo de detección
#modelo_deteccion = load_efficientdet_model()

for img_archivo in img_archivos:
    img_ruta = os.path.join(img_carpeta, img_archivo)

    # Cargar y preprocesar la imagen
    img = load_image(img_ruta)
    img_preprocesada = preprocess_image(img)

    # Detectar las frutillas
    #detecciones = detect_fruit(img_preprocesada, modelo_deteccion)

    # Visualizar los resultados
    #img_con_detecciones = draw_detections(img, detecciones, ["Frutilla"])
    display_image(img_preprocesada)