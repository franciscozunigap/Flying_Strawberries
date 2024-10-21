import os
from modules.detection import load_model, detect_strawberries
from modules.utils import display_results, save_predicts

# Cargar los directorios de imágenes
proyect_dir = os.getcwd()
input_images_dir = os.path.join(proyect_dir, "img/input_predict")
output_predict_dir = os.path.join(proyect_dir, "img/output_predict")

# Cargar el modelo de detección
modelo_deteccion = load_model()

# Detectar las frutillas
resultados = detect_strawberries(modelo_deteccion, input_images_dir)
print(f"Se procesaron {len(resultados)} imágenes.")

# Visualizar las imágenes
display_results(resultados)

# Guardar las imágenes
save_predicts(resultados, output_predict_dir)