from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('trained_model.pt')

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

    # Get both segmentation masks and detection boxes
    results = model(source=frame, save=False)
    
    for result in results:
        # Process segmentation masks
        masks = result.masks
        if masks is not None:
            for mask in masks:
                mask = mask.data.cpu().numpy()          # Convert to numpy array
                mask = (mask * 255).astype(np.uint8)    # Scaling image from [0,1] to [0,255] and converting it to 8 bits
                mask = mask.squeeze(axis=0)             # [1,x,y] => [x,y]
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resizing
                colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Converting from grayscale to BGR
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.3, 0)

        # Process detection boxes
        boxes = result.boxes
        if boxes is not None:
            max_box = None
            area = 0
            for box in boxes:
                # Get box coordinates and class ID
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                confidence = box.conf[0].cpu().numpy()

                if((x2-x1)*(y2-y1) > area) :
                    area = (x2-x1)*(y2-y1)
                    max_box = (x1, y1, x2, y2, class_id, confidence)

            # Draw the bounding box for the largest box
            x1, y1, x2, y2, class_id, confidence = max_box
            label = f'{model.names[class_id]} {confidence:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mid = (x1+x2)//2
            cv2.line(frame, (mid, y1), (mid, y2), (255, 0, 0), 2)  
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Segmentation and Object Detection', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
