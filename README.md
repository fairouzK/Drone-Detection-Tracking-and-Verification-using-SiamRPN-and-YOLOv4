# Drone-Detection-Tracking-and-Verification-using-SiamRPN-and-YOLOv4

This repo was based on and refined from these two repos:

1: Updated siamRPN for multiple object tracking: https://github.com/adwardlee/multi-obj-tracking-siam.git

2: Drone verification using siamRPN for control signals and control signal calculation: https://github.com/ghimireadarsh/Drone-Verification-using-SiamRPN-Tracker.git

## How to run
This project is developed using anaconda. The installation steps are described below.
1. install packages in a conda virtual env

  conda install pytorch torchvision -c pytorch
  
  pip install opencv-python imutils pyyaml yacs tqdm colorama matplotlib cython tensorboardX

2. create a 'backup' folder in darknet folder and download the yolov4 weights to the same folder.
3. run using 'python drone_detector_and_tracker_final.py --boxnum 2' to track 2 objects. 


## YOLOv4
Custom dataset was downloaded from kaggle: https://www.kaggle.com/dasmehdixtr/drone-dataset-uav
steps to traning yolo with custom dataset can be found here: 
https://github.com/AlexeyAB/darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects 
or 
https://towardsdatascience.com/installing-ubuntu-20-04-lts-and-running-yolov4-and-yolov5-on-it-2ca0c93e244a


## Reference
