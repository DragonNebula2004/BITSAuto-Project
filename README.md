# Overview:
In this project, we implement the YOLOv8 segmentation model. To get a basic understanding of YOLO, you can go through the following playlist [yolo videos](https://www.youtube.com/playlist?list=PL1u-h-YIOL0sZJsku-vq7cUGbqDEeDK0a).
You can also go through the documentation [YOLOv8](https://docs.ultralytics.com/models/yolov8/). We train this model on a custom dataset comprising of the roads in Bits Goa. We used [Roboflow](https://roboflow.com/) to annotate the dataset.

# Requirements
- ultralytics
- opencv-python
- numpy
  
command to install the requirements:
```bash
pip install <requirement>
```
# File description
train_model.py : We train a pre-trained model on our custom dataset and save it. To run this file, you would have to download the datasets folder.

yolo_segment.py : Implemention of our model.

yolo_detect_segment.py : In this file, apart from segmentation we draw bounding boxes around the road in an attempt to find its midpoint.

To run the latter 2 files, you would have to download the video, IMG_3010.mp4. 

Google drive link for the downloads : [dataset](https://drive.google.com/drive/folders/1_MA48VKG8hAU8YRUScmKIu83-DeEHZZM?usp=drive_link).


# Future Work
Marking the left and right boundaries of the road.  
Dividing the road into 2 lanes.  
Exploring other models like SAM etc.  




