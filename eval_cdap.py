# This script exists to download the cadp dataset

# Imports
import json
import os
import os.path as path
import cv2 as cv
import torchvision.transforms
from ultralytics import YOLO
import torch
from super_image import EdsrModel, ImageLoader
from PIL import Image
from torchvision.transforms import v2 as tf
import numpy as np
from yolot.BMOTSDataset import class_dict, label_dict

# Constants
annotation_filepath = "C:\\Users\\Dylan\\Documents\\Data\\CADP\\annotations_1531762138.1303267.json"
video_dst = "videos"
model_name = "yolo_nas"

def main_func():
    # initialize text settings
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_color = (0, 255, 255)  # Text color in BGR format
    thickness = 1
    position = (20, 20)

    # Upscaling
    #sr_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    #sr_model.to(0)
    #sr_model.eval()


    # Transforms
    transforms = tf.Compose([
        tf.Resize(size=(640,640)),
    ])

    # Read in annotations
    with open(annotation_filepath) as file:
        data = json.load(file)

    # Initialize model
    # Build Baseline
    baseline = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\test_run_big_data\\weights\\best.pt"
    model = YOLO(baseline, task="predict")
    model.to(0)

    # Create Writer
    output_size = (720,1280)
    save_path = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\yolot\\val_runs\\cadp_baseline.mp4"
    writer = create_writer(save_path, output_size)

    # Display the accident segments of each video annotation
    exiting = False
    for vid in data:
        # Create video reader with selected clip
        cap = cv.VideoCapture(os.path.join("C:\\Users\\dylan\\Documents\\Data\\CADP\\forth_investigation", vid))
        h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv.CAP_PROP_FPS)
        # Skip low resolution images
        if w >= 720 and h >= 720:
            for idx, seg in enumerate(data[vid]):
                start_time_ms = seg['keyframes'][0]['frame']*1000
                end_time_ms = seg['keyframes'][1]['frame']*1000
                # Jump to the start of the accident clip
                cap.set(cv.CAP_PROP_POS_MSEC, start_time_ms)
                # Read Frames until end is reached
                ret = True
                while cap.get(cv.CAP_PROP_POS_MSEC) < end_time_ms and ret==True:
                    # Read frame
                    ret, frame = cap.read()
                    # If frame is valid
                    if ret:
                        # Run YOLO model
                        dets = model.predict(transforms(frame), conf=0.5)

                        # Display Baseline
                        base_boxes = dets[0].boxes.xyxy.to(int)
                        base_cls = dets[0].boxes.cls.to(int)
                        base_conf = dets[0].boxes.conf.to(float) 
                        for det_idx in range(base_boxes.shape[0]):
                            conf = float(base_conf[det_idx])
                            cls = int(base_cls[det_idx])
                            x1 = int(base_boxes[det_idx][0])
                            y1 = int(base_boxes[det_idx][1])
                            x2 = int(base_boxes[det_idx][2])
                            y2 = int(base_boxes[det_idx][3])
                            cv.rectangle(frame,(x1,y1),(x2,y2), (0,0,150), thickness=2)
                            write_label(frame, x1, y1,x2,y2, f"{label_dict[cls]}:{conf:.2f}", color=(0,0,255), mode='bl')

                        # Write information
                        cv.putText(frame, f"{vid}-{idx}",
                                   position,
                                   fontScale=font_scale,
                                   color=font_color,
                                   thickness=thickness,
                                   fontFace=font)
                        cv.putText(frame, f"{start_time_ms}-{end_time_ms}",
                                   (0, int(h - cv.getTextSize(f"{start_time_ms}-{end_time_ms}", font, font_scale,
                                                              thickness)[0][1])),
                                   fontScale=font_scale * 0.5,
                                   color=font_color,
                                   thickness=thickness,
                                   fontFace=font)
                        # Create Plate Frame
                        frame = place_center(frame, output_size)
                        # Display Frame
                        cv.imshow("Dataset Cam",frame)
                        # Write to video
                        writer.write(frame)
                        # Delay
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            exiting = True
                            break
                    else:
                        print(f"Unable to read frame {vid} - {idx} - {start_time_ms} - {end_time_ms}")
                if exiting:
                    break
        # Release Cap Object and
        cap.release()
        if exiting:
            break

    writer.release()
    print("Done!")

def place_center(img, size):
    # Create a blank image to place the input image on
    plate = np.zeros(shape=[size[0],size[1], 3], dtype=np.uint8)
    pcx = int(size[0]/2)
    pcy = int(size[1]/2)
    # Place the image in the center
    w = img.shape[0]
    h = img.shape[1]
    plate[pcx-int(w/2):pcx+int(w/2),pcy-int(h/2):pcy+int(h/2)] = img
    return plate

def create_writer(filename, size, fps=30):
    # Define the output file name, codec, FPS, frame size, and color flag
    output_file = filename
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for AVI format
    fps = 30  # Frames per second
    frame_size = (size[1], size[0])  # Frame size (width, height)
    is_color = True  # True for color, False for grayscale

    # Create a VideoWriter object
    return cv.VideoWriter(output_file, cv.CAP_FFMPEG, fourcc, fps, frame_size, is_color)

def write_label(image,x1,y1,x2,y2,label, color=(255,255,255), mode='tl'):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    label_size, baseline = cv.getTextSize(label, font, font_scale, font_thickness)
    if mode == 'bl':
        cv.putText(image, label, (x1, y2 + 2*baseline), font, font_scale, color, font_thickness, cv.LINE_AA)
    elif mode == 'ct':
        cv.putText(image, label, (x1, y2 - baseline), font, font_scale, color, font_thickness, cv.LINE_AA)
    else:
        cv.putText(image, label, (x1, y1 - baseline), font, font_scale, color, font_thickness, cv.LINE_AA)

if __name__ == '__main__':
    main_func()