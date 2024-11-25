from ultralytics import YOLO

def load_detect_model(model_path = "runs/detect/train/weights/best.pt"):
    model = YOLO(model_path)
    return model

def load_classify_model(model_path = "runs/classify/train/weights/best.pt"):
    model = YOLO(model_path)
    return model