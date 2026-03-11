# Overview:
In this project, we implement the YOLOv8 segmentation model. To get a basic understanding of YOLO, you can go through the following playlist [yolo videos](https://www.youtube.com/playlist?list=PL1u-h-YIOL0sZJsku-vq7cUGbqDEeDK0a).
You can also go through the documentation [YOLOv8](https://docs.ultralytics.com/models/yolov8/). We train this model on a custom dataset comprising of the roads in Bits Goa. We used [Roboflow](https://roboflow.com/) to annotate the dataset.

# Requirements
- ultralytics
- opencv-python
- numpy
  

# File description
train_model.py : We fine tune a pre-trained model on our custom dataset and save it. To run this file, you would have to download the datasets folder.

yolo_segment.py : Implemention of our model.

yolo_detect_segment.py : Segmentation + Bounding Boxes around the road in an attempt to find its midpoint.

To run the latter 2 files download the video, IMG_3010.mp4. 

Google drive link for the downloads : [dataset](https://drive.google.com/drive/folders/1_MA48VKG8hAU8YRUScmKIu83-DeEHZZM?usp=drive_link).




