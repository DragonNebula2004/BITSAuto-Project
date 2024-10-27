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
    if not ret :
        break
    frame_count += 1
    if(frame_count%skip_frame != 0):
        continue
    results = model(source=frame,save=False)
    for result in results:
        masks = result.masks
        for mask in masks:
            mask = mask.data.cpu().numpy()          # Convert to numpy array
            mask = (mask * 255).astype(np.uint8)    # scaling img from [0,1] to [0,255] and converting it to 8bits
            mask = mask.squeeze(axis=0)             # [1,x,y] => [x,y]
            mask = cv2.resize(mask,(frame.shape[1],frame.shape[0])) #resizing
            #print(mask.shape)
            #print(frame.shape)
            colored_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR) #converting from grayscale to bgr
            frame = cv2.addWeighted(frame, 1, colored_mask, 0.3, 0)
    cv2.imshow('YOLOv8 Segmentation', frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()
