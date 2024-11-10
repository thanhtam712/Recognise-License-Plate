from ultralytics import YOLOv10


def load_model_yolov10():
    model_yolov10 = YOLOv10(
        "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/yolov10/runs/detect/train3/weights/last.pt"
    )
    return model_yolov10


def detect_yolov10(model_yolov10, input: str):
    results = model_yolov10.predict(source=input, save=True, classes=[0])
    return results
