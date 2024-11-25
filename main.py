from modules.models import load_detect_model, load_classify_model
from modules.algoritm import inference

# Cargar los modelos
detect_model = load_detect_model()
classify_model = load_classify_model()

# Ruta de la imagen a procesar
image_path = "img/R.png"

inference(image_path, detect_model, classify_model)