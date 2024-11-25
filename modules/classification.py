import os
from ultralytics import YOLO

def load_classify_model(model_path = os.path.join(os.getcwd(), "runs/classify/train/weights/best.pt")):
    model = YOLO(model_path)
    return model