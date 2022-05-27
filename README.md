## Drone-Detection-Tracking-and-Verification-using-SiamRPN-and-YOLOv4

This repo was based on and refined from these two repos:

1: Updated siamRPN for multiple object tracking: https://github.com/adwardlee/multi-obj-tracking-siam.git

2: Drone verification using siamRPN for control signals and control signal calculation: https://github.com/ghimireadarsh/Drone-Verification-using-SiamRPN-Tracker.git

### How to run
This project is developed using anaconda. The installation steps are described below.
1.	install packages in a conda virtual env
<pre><code> 
conda install pytorch torchvision -c pytorch
pip install opencv-python imutils pyyaml yacs tqdm colorama matplotlib cython tensorboardX
</code></pre>
2.	create a 'backup' folder in darknet folder and download the yolov4 weights to the same folder. yolo weights can be downloaded <a href = "https://drive.google.com/file/d/1nO6Lxg5QrMmcNM2gwWyIc3RPcXpeXw3x/view?usp=sharing"> here. </a>
3.	To track objects, run using 

one object: <pre><code> python drone_detector_and_tracker_final.py </code></pre>
multiple objects: <pre><code> python drone_detector_and_tracker_final.py --boxnum 2 </code></pre> 

### YOLOv4
Custom dataset was downloaded from <a href="https://www.kaggle.com/dasmehdixtr/drone-dataset-uav">Kaggle</a>
Steps to training yolo with custom dataset can be found from <a href="https://github.com/AlexeyAB/darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects"> AlexeyAb repo</a> or <a href="https://towardsdatascience.com/installing-ubuntu-20-04-lts-and-running-yolov4-and-yolov5-on-it-2ca0c93e244a"> This blog </a>

### Results
##### Detection and Tracking
<img src= "/results/Picture1.png" width = "250" />

##### Drone 0 moves vertically
<img src= "/results/Picture2.png" width = "250" />

##### Drones classified as Good or Bad
<img src= "/results/Picture3.png" width = "250" />

##### Printed results
<img src= "/results/Picture4.png" width = "250" />

## Reference
