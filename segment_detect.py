from ultralytics import YOLO
import numpy as np
import cv2

# Load YOLO models
segmentation_model = YOLO('trained_model.pt')  # Your custom road segmentation model
detection_model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 object detection model (COCO dataset)

video_path = "IMG_3010.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

skip_frame = 20
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % skip_frame != 0:
        continue

    # Road segmentation
    seg_results = segmentation_model(source=frame, save=False)
    for seg_result in seg_results:
        masks = seg_result.masks
        if masks is not None:
            for mask in masks:
                mask = mask.data.cpu().numpy()          # Convert to numpy array
                mask = (mask * 255).astype(np.uint8)    # Scale to [0,255]
                mask = mask.squeeze(axis=0)             # [1,x,y] -> [x,y]
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize
                colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.3, 0)  # Overlay mask

        # Process road bounding box
        x1_max, y1_max, x2_max, y2_max = None, None, None, None
        boxes = seg_result.boxes
        if boxes is not None:
            max_box = None
            area = 0
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                confidence = box.conf[0].cpu().numpy()

                if (x2 - x1) * (y2 - y1) > area:
                    area = (x2 - x1) * (y2 - y1)
                    max_box = (x1, y1, x2, y2, class_id, confidence)

            if max_box:
                x1_max, y1_max, x2_max, y2_max, class_id, confidence = max_box
                label = f'{segmentation_model.names[class_id]} {confidence:.2f}'
                cv2.rectangle(frame, (x1_max, y1_max), (x2_max, y2_max), (0, 255, 0), 2)
                
                # Draw center lane divider
                mid_x = (x1_max + x2_max) // 2
                cv2.line(frame, (mid_x, y1_max), (mid_x, y2_max), (255, 0, 0), 2)  

                cv2.putText(frame, label, (x1_max, y1_max - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Object detection (COCO dataset)
    obj_results = detection_model(source=frame, save=False)
    for obj_result in obj_results:
        obj_boxes = obj_result.boxes
        if obj_boxes is not None:
            for obj_box in obj_boxes:
                x1, y1, x2, y2 = map(int, obj_box.xyxy[0].cpu().numpy())
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # Only mark objects on the road
                if not ((x1_max <= center_x <= x2_max) and (y1_max <= center_y <= y2_max)) :
                    continue
                class_id = int(obj_box.cls[0].cpu().numpy())
                confidence = obj_box.conf[0].cpu().numpy()
                label = f'{detection_model.names[class_id]} {confidence:.2f}'

                # Draw bounding boxes for detected objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Segmentation and Object Detection', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
