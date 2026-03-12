from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO("yolov8s-cls.pt")

# Train
model.train(
    data="/home/abhay/yolo_cls_forus",
    epochs=50,
    imgsz=224,
    batch=16
)
