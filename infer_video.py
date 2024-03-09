'''
    File: test_yolot
    Author: Dylan Baxter
    Desc: This script is intended to be used to run trained yolot models and inspect output
    Date Authored: 9/13/23
'''

# Imports
import numpy as np
import torch
import cv2
import argparse
import os
from os import path
from torchvision.transforms import ToTensor
from tqdm import tqdm
import datetime

from yolot.SequenceModel import SequenceModel
from ultralytics.utils.ops import non_max_suppression
from yolot.BMOTSDataset import class_dict, label_dict

from ultralytics import YOLO


# Defaults and Macros
default_model_path = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\med_gru\\weights\\best.pth"
default_save_dir = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\yolot\\val_runs"
default_vid_path = "med_gru_11"
default_data_path = "C:\\Users\\dylan\\Documents\\Data\\sample_videos\\1.mp4"
default_model_conf = "cfg/models/yolot_gru_bigm.yaml"
default_device = 0
FRAME_RATE = 30.0

# Options Parser
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to model to be trained')
    parser.add_argument('--run_name', type=str, default=default_vid_path, help='Path to inference video ')
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default=default_save_dir, help="Path to which visualization will be saved")
    parser.add_argument('--save_name', type=str, default="output.mp4", help="Path to which visualization will be saved")
    parser.add_argument('--data', type=str, default=default_data_path, help="Path to Dataset")
    parser.add_argument('--device', type=int, default=default_device, help="device")
    parser.add_argument('--model_cfg', type=str, default=default_model_conf, help="model config")
    parser.add_argument('--iou', type=float, default=0.25)
    parser.add_argument('--conf', type=float, default=0.3)
    return parser

# Main Func
def main_func(args):
    # Establish args
    model_path = args.model_path
    model_cfg = args.model_cfg
    data_path = args.data
    DISP = args.disp
    save_path = args.save_path
    device = args.device
    save_name = args.save_name
    nms_iou = args.iou
    nms_conf = args.conf


    # Build Model
    model = SequenceModel(cfg=model_cfg, device=0, verbose=False)
    model.eval()
    # Load Weights
    model.load_state_dict(torch.load(model_path)['model'])
    model.model_to(0)

    # Setup Video Writer
    if save_path:
        writer = cv2.VideoWriter(os.path.join(save_path, save_name), 
                                 cv2.VideoWriter_fourcc(*'mp4v'), 
                                 FRAME_RATE, (640, 640))
        
    cap = cv2.VideoCapture(data_path)

    # Iterate through the validation dataset
    total_detections = 0
    seq_idx = 0
    ret = True
    stop = False
    model.zero_states()
    while not stop:
        ret, im = cap.read()
        if ret:
            im = cv2.resize(im, (640,640))
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_torch = torch.from_numpy(im_rgb.astype('float32')/255.0).permute(2, 0, 1).unsqueeze(0)

            start_time = datetime.datetime.now()
            with torch.no_grad():
                outputs = model(im_torch)
            inf_time = datetime.datetime.now() - start_time

            for frame_idx in range(im_torch.shape[0]):
                # Preprocess Frame for OpenCV
                frame = im
                # Process Detections
                raw_dets = outputs[frame_idx][0]
                # NMS
                dets = non_max_suppression(raw_dets, conf_thres=nms_conf, iou_thres=nms_iou)
                print(f"Frame {frame_idx}: {dets[0].shape[0]} detections, inf: {inf_time.microseconds/1000}")
                for det_idx in range(dets[0].shape[0]):
                    conf = float(dets[0][det_idx,4])
                    cls = int(dets[0][det_idx, 5])
                    x1 = int(dets[0][det_idx,0])
                    y1 = int(dets[0][det_idx,1])
                    x2 = int(dets[0][det_idx,2])
                    y2 = int(dets[0][det_idx,3])
                    #x2 = x1+w
                    #y2 = y1+h
                    cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0))
                    write_label(frame, x1, y1, f"{label_dict[cls]}:{conf:.2f}")
                    total_detections += 1

                # Show Image
                cv2.imshow("Predictions", frame)
                writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop = True
                    break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            break
        
        seq_idx += 1
    
    writer.release()
    print("Inference Complete!")

def write_label(image,x1,y1,label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.putText(image, label, (x1, y1 - baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

''' Main Script'''
if __name__ == '__main__':
    args = init_parser().parse_args()
    print(args)
    main_func(args)
