from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
model.train(data="datasets/data.yaml",epochs=5)

model.save('trained_model.pt')