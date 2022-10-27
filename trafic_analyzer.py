
from collections import OrderedDict
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import sys
from centroidtracker import CentroidTracker


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default=None,
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--confidence", type=float, default=.85,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def project_pt(image, point):
    
    image_h, image_w, __ = image.shape
    x = int((point[0]/darknet_width) * image_w)
    y = int((point[1]/darknet_height) * image_h)

    return x,y



def video_capture(frame_queue, tracking_img_queue):
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)

        tracking_img_queue.put(frame_resized)
        frame_queue.put(frame)

    cap.release()
    os._exit(1)




def generate_analytics(frame, tobjects):
    global inbound_count
    global outbound_count
    global IDs_counted
    global class_counts
    global crossing_counted

    analytics_text = []

    for tid, tobj in tobjects.items():
        
        cur_pos = project_pt(frame, tobj.cur_position)
        starting_pos = project_pt(frame, tobj.starting_position)

        if(tid not in IDs_counted):
            class_counts[tobj.label]+=1
            IDs_counted.add(tid)
        
        if(starting_pos[1] < Boundary_Line[1]):  #if the object was detected above boundary line
             if(cur_pos[1] > Boundary_Line[1]): # if object has crossed the boundary line
                 if(tobj.ID not in crossing_counted):
                    inbound_count+=1
                    crossing_counted.add(tobj.ID)
        else:   #if the object was detected below boundary
            if(cur_pos[1] < Boundary_Line[1]): #if the object has moved above the boundary line
                if(tobj.ID not in crossing_counted):
                    outbound_count+=1
                    crossing_counted.add(tobj.ID)

        

    analytics_text.append("Inbound: {}".format(inbound_count) )
    analytics_text.append("Outbound: {}".format(outbound_count))

    for clabel in OOI:    # gather counts for each class type
        analytics_text.append("{}:{} ".format(clabel,class_counts[clabel]) )
        
    return analytics_text


def detect_and_track(tracking_img_queue, detection_queue , tracker_queue ,fps_queue):
    
    ct = CentroidTracker()

    while cap.isOpened():

        tracking_image = tracking_img_queue.get()
        prev_time = time.time()
        

        darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(darknet_image, tracking_image.tobytes())
        
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)

        darknet.print_detections(detections, args.ext_output)
        
        bboxes = []
        rects = []
        confidences = []
        detections_filtered = []
        labels = []
        # Create a new tracker for each detection
        for label, confidence, bbox in detections:

            if(float(confidence) > args.confidence and (label in OOI)): #If confidence of detection is high and the object is of interest save that object
                (centerX, centerY, width, height) = bbox
                bboxes.append(bbox)
                rect = [int(centerX) - int(width/2), int(centerY) - int(height/2), int(centerX) + int(width/2), int(centerY) + int(height/2)]
                rects.append( rect )
                confidences.append(float(confidence))
                labels.append(label)
                detections_filtered.append((label,confidence,bbox))
        detection_queue.put(detections_filtered)

        rects_filtered = []
        labels_filtered = []
        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(bboxes, confidences, args.confidence,args.thresh)
        if len(idxs) > 0:
            for i in idxs.flatten():
                rects_filtered.append(rects[i])
                labels_filtered.append(labels[i])

        Objects  = ct.update(labels_filtered,rects_filtered)
        tracker_queue.put(Objects)
        darknet.free_image(darknet_image)

        total_time = (time.time() - prev_time)
        fps = int(1/(total_time))
        fps_queue.put(fps)

    cap.release()
    os._exit(1)


def drawing(frame_queue, detection_queue ,tracker_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    if(args.out_filename is not None):
        video = set_saved_video(cap, args.out_filename,
                                (video_width, video_height))

    while cap.isOpened():
        frame = frame_queue.get()

        fps = fps_queue.get()
        print("FPS: {}".format(fps))

        detections_adjusted = []
        if frame is not None:
            # print("Drawing {} object rectangles".format(len(tracker_output_queues)))
            objects = tracker_queue.get()
            detections = detection_queue.get()
            
            
            for tid, tobj in objects.items():
                 
                text = "ID {}".format(tid)
                centroid = tobj.cur_position
                cx,cy = project_pt(frame,centroid)
                
                cv2.putText(frame, text, (cx - 10, cy - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            for label,confidence,bbox in detections:
                
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            frame = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            
            #drawing line for counter
            frame = cv2.line(frame, (0,Boundary_Line[1]), (Boundary_Line[0],Boundary_Line[1]), (255,255,0), 5)
            #Adding analytics to the frame
            analytics_text = generate_analytics(frame, objects)
            for i in range(len(analytics_text)):
                cv2.putText(frame, analytics_text[i], (10, 20+i*20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            if not args.dont_show:
                cv2.imshow('Tracking', frame)
            if args.out_filename is not None:
                video.write(frame)
            if cv2.waitKey(20) == 27:
                os._exit(1)
                # break

    cap.release()
    if(args.out_filename is not None):
        video.release()
    cv2.destroyAllWindows()
    os._exit(1)


if __name__ == '__main__':

    # objects of interest
    OOI = ['person','bicycle','car','bus']
    class_counts = {'person':0,'bicycle':0,'car':0,'bus':0}
    Boundary_Line = (1280,300)
    inbound_count = 0
    outbound_count = 0
    IDs_counted = set()
    crossing_counted = set()

    frame_queue = Queue(maxsize=1)
    detection_queue = Queue(maxsize=1)
    tracker_queue = Queue(maxsize=1)
    tracking_img_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )


    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Thread(target=video_capture, args=(
        frame_queue, tracking_img_queue)).start()
    Thread(target=detect_and_track, args=(
        tracking_img_queue, detection_queue, tracker_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detection_queue,
                                 tracker_queue, fps_queue,)).start()
