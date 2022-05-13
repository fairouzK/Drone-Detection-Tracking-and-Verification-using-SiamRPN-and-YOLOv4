import glob
import torch
from multiobjTrackingSiam.siamrpn import TrackerSiamRPN
import re
from imutils.video import VideoStream
from multiobjTrackingSiam.util import *
from imutils.video import FPS
import os
import argparse
import math
import numpy as np
import cv2
import torch
import imutils
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--base_path', default='video_frames/', help='datasets')
parser.add_argument('--net_path', type = str, default='multiobjTrackingSiam/experiments/siamrpn_alex_dwxcorr/model.pth', help='network path')
parser.add_argument('--object', nargs='*',action="store",help='set object region') 
parser.add_argument('--region', nargs='*',action="store",help='set alarm region')
parser.add_argument('-v','--video',type = str, default = 'demo_vid.mp4', help='video name')
parser.add_argument('--type', type = int, default = 0, help = 'tracker type')
parser.add_argument('--boxnum', type = int, default = 2, help = 'box number')

args = parser.parse_args()

def main():

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.backends.cudnn.benchmark = True

    ### init model ###
    tracker = TrackerSiamRPN(net_path=args.net_path)

    # initialize the bounding box coordinates of the object we are going to track
    gt_bbox = None

    if not (args.video == False):
        print("[INFO] starting video stream...")
        cap = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        cap = cv2.VideoCapture(args.video) # might not be necessary

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'avc1') #(*'MP42')
    #outputV = cv2.VideoWriter('output-CPU.mp4', fourcc, cv2.CAP_PROP_FPS, (640, 480))

    toc = 0
    f = 0    

    frame = cap.read() # first frame
    
    # Load the yolo weights and cfg file
    weights_path = "darknet/backup/yolo-obj_final.weights" 
    cfg_path = "darknet/cfg/yolo-obj.cfg" 
   
    # If CUDA is available
    net = cv2.dnn.readNet(weights_path, cfg_path)
    useCuda = False
    if useCuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # Ask network to use specific computation backend where it supported.
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)   # Ask network to make computations on specific target device. 
    
    # Get the class names 
    filename =  "darknet/data/obj.names" #"darknet/data/coco.names"
    classes = []
    with open(filename, "r") as fil:
        classes = [line.strip() for line in fil.readlines()]
    
    # Get height and width of frame
    height, width, channels = frame.shape 
   
    # Detect the objects (drones in the frame)
    object = get_prediction(net, frame, classes, width, height, args.boxnum)  
    if len(object) < args.boxnum:   # If number of detected drones is less than specified, update boxnum argument 
        args.boxnum = len(object)
        print("Box num: ", len(object))
        gt_bbox = 1

    drone_x = [[] for _ in range(args.boxnum)] # drones movement in x before control signal is sent
    drone_y = [[] for _ in range(args.boxnum)] # drones movement in y before control signal is sent
    after_command_x = [[] for _ in range(args.boxnum)] # drones movement in x after control signal is sent
    after_command_y = [[] for _ in range(args.boxnum)] # drones movement in y after control signal is sent
    control_data = []  # The m component of control data calculated
    control_data_1 = [] # The b component of control data calculated
    drone_status = []  # List to store the updated state of the drones
    success_arr = []  # Number of successful tracker initiations
    verification_threshold = 20 # in degrees for looking out how much angle does the drone changes
    limit_datapoint = 50
    control_signal_status = []
    command_exec_time = True # to track drone within certain time after sending control signal
    execution_deadline = 100
    drone_success = [[] for _ in range(args.boxnum)]  
    drone_color = []  # bounding box colors 
    while True:
        frame = cap.read()   
        frame = frame[1] if (args.video == False) else frame
        
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        
        tic = cv2.getTickCount()

        if gt_bbox is not None:
            if f == 0:  # the first frame           
                tracker.init(frame, object)  # Initialize trakcer for every drone
                for i in range(args.boxnum):
                    drone_color.append((255, 0, 0))  
                
            if f > 0:  # tracking
                box = tracker.update(frame) ### x - w / 2,y - h /2,w,h ### (number, 4)
                for idx in range(box.shape[0]):
                    #print("Idx = {} and detected = {}".format(idx, box.shape[0]))
                    x,y,w,h = box[idx].tolist()
                    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, drone_color[idx], 3)
                    success_arr.append(True)
                    drone_status.append("detected")

                key = cv2.waitKey(1)
                if key > 0:
                    break
            f += 1
            toc += cv2.getTickCount() - tic
        
        cv2.imshow("Frame", frame)
        #outputV.write(frame)
        key = cv2.waitKey(1) & 0xFF

        if (len(success_arr) != 0) and (len(control_data) == 0):
            for pos in range(args.boxnum):
                if len(drone_x[pos]) < limit_datapoint:
                        drone_x[pos].append(box[pos][0]+box[pos][2]/2)
                        drone_y[pos].append(box[pos][1]+box[pos][3]/2)
                else:
                    print("Total points of drone {}: {}".format(pos, len(drone_x[pos])))
                    # control signal function
                    best_fit = np.polyfit(drone_x[pos], drone_y[pos], deg=1)  # returns (m, b)
                    (control_d, control_d1) = control_signal(best_fit)
                    control_data.append(control_d)
                    control_data_1.append(control_d1)
                    print("Best fit : {}".format(best_fit))
                    print("Theta : {}".format(math.degrees(math.atan(best_fit[0]))))
                    print("Control signal data received : {}".format(control_data[pos]))
                    control_signal_status.append("Complete")
       
        if (len(success_arr) != 0) and (len(control_data) != 0) and command_exec_time:
        #if success and control_data is not None and command_exec_time:
            for pos in range(args.boxnum):
                drone_status[pos] = "VERIFYING"
                if len(after_command_x[pos]) < limit_datapoint:
                    after_command_x[pos].append(box[pos][0] + box[pos][2] / 2)
                    after_command_y[pos].append(box[pos][1] + box[pos][3] / 2)
                else:
                    after_command_best_fit = np.polyfit(after_command_x[pos], after_command_y[pos], deg=1)  # returns (m, b)
                    after_command_theta = math.degrees(math.atan(after_command_best_fit[0]))
                    
                    diff = abs(abs(after_command_theta) - abs(control_data[pos]))
                    diff_1 = abs(abs(after_command_theta) - abs(control_data_1[pos]))

                    # here time limit to be added later
                    if diff < verification_threshold or diff_1 < verification_threshold:
                        drone_success[pos].append(1)
                    else:
                        drone_success[pos].append(0)
                        after_command_x[pos].pop(0)
                        after_command_y[pos].pop(0)
                        after_command_x[pos].append(box[pos][0]+box[pos][2]/2)
                        after_command_y[pos].append(box[pos][1]+box[pos][3]/2)

        for pos in range(args.boxnum):
            if len(drone_success[pos]) > execution_deadline:
                command_exec_time = False
                if drone_success[pos].count(1)/len(drone_success[pos]) >= 0.2:
                    print("***************Good drone {}****************".format(pos))
                    drone_status[pos] = "GOOD"
                    drone_color[pos] = (0, 255, 0)
                else:
                    print("***************Bad drone {}*****************".format(pos))
                    drone_status[pos] = "BAD"
                    drone_color[pos] = (0, 0, 255)
                
        if key == ord("q"): 
            break

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    cap.release()
    #outputV.release()
    cv2.destroyAllWindows()


def get_prediction(net, frame, classes, width, height, numbbox):

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[0 - 1] for i in net.getUnconnectedOutLayers()]
    font = cv2.FONT_HERSHEY_PLAIN
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    bounding_box = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                              
                boxes.append([x, y, w, h])  
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if (label == "Drone"):    # Id an object of interest only
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
                if(len(bounding_box) < numbbox):
                    bounding_box.append([x, y, h, w])
    return bounding_box

def control_signal(best_fit):
    theta = math.degrees(math.atan(best_fit[0]))
    control_d = theta + 90
    control_d1 = theta - 90
    start = time.time()
    while(time.time()-start < 0.3):
        print("Sending control signal, theta {}".format(control_d))
        time.sleep(0.1)
    return (control_d, control_d1) # for positive and negative angle
    
if __name__ == '__main__':
    main()