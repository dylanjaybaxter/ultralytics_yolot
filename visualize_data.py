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
from ultralytics.data.build import InfiniteDataLoader
from yolot.SequenceModel import SequenceModel
from yolot.BMOTSDataset import BMOTSDataset, collate_fn, single_batch_collate
from ultralytics.utils.ops import non_max_suppression
from yolot.val import SequenceValidator
from yolot.BMOTSDataset import class_dict, label_dict

from ultralytics import YOLO


# Defaults and Macros
default_model_path = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\blank_check\\weights\\last.pt"
default_base = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\test_run_big_data\\weights\\best.pt"
default_save_dir = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\yolot\\val_runs"
default_vid_name = "test.mp4"
default_data_path = "C:\\Users\\dylan\\Documents\\Data\\BDD100k_MOT202\\bdd100k"
default_model_conf = "cfg/models/yolotn.yaml"
default_device = 0
FRAME_RATE = 10.0

# Options Parser
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to model to be trained')
    parser.add_argument('--base', type=str, default=default_base, help='Path to model to be trained')
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default=default_save_dir, help="Path to which visualization will be saved")
    parser.add_argument('--save_name', type=str, default=default_vid_name, help="Path to which visualization will be saved")
    parser.add_argument('--data', type=str, default=default_data_path, help="Path to Dataset")
    parser.add_argument('--device', type=int, default=default_device, help="device")
    parser.add_argument('--model_cfg', type=str, default=default_model_conf, help="model config")
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--conf', type=float, default=0.25)
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
    baseline = args.base

    # Set up val dataloader
    print("Building Dataset...")
    val_dataset = BMOTSDataset(data_path, "val", device=0, seq_len=100, data_cap=1000, shuffle=False, aug=True, drop=0.0, mixup=2)
    val_loader = InfiniteDataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, drop_last=False, pin_memory=False)

    # Setup Video Writer
    if save_path:
        writer = cv2.VideoWriter(os.path.join(save_path, save_name), 
                                 cv2.VideoWriter_fourcc(*'mp4v'), 
                                 FRAME_RATE, (640, 640))

    # Compile
    acc = 0
    running_acc = 0
    bar_format = f"::Validation | {{bar:30}}| {{percentage:.2f}}% | [{{elapsed}}<{{remaining}}] | {{desc}}"
    pbar_desc = f"Seq:.../..., Loss: {acc:.10e}, lr:{running_acc:.5e}"
    pbar = tqdm(val_loader, desc=pbar_desc, bar_format=bar_format, ascii=False)

    # Iterate through the validation dataset
    total_detections = 0
    stop = False
    for seq_idx, subsequence in enumerate(pbar):
        sub_ims = subsequence['img']

        for frame_idx in range(sub_ims.shape[0]):

            # Preprocess Frame for OpenCV
            frame = sub_ims[frame_idx,:,:,:].cpu().numpy()
            frame = (np.transpose(frame[0,:,:,:], (1,2,0))*255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Get Ground Truth
            gt_boxes = subsequence['bboxes'][subsequence['batch_idx']==frame_idx]
            gt_cls = subsequence['cls'][subsequence['batch_idx']==frame_idx]

            # Display Ground Truth
            for det_idx in range(gt_boxes.shape[0]):
                cls = int(gt_cls[det_idx])
                w = int(gt_boxes[det_idx,2]*640)
                h = int(gt_boxes[det_idx,3]*640)
                x1 = int(gt_boxes[det_idx,0]*640) - int(w/2)
                y1 = int(gt_boxes[det_idx,1]*640) - int(h/2)
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(frame,(x1,y1),(x2,y2), (200,200,200), thickness=1)
                write_label(frame, x1, y1,x2,y2, f"{label_dict[cls]}", color=(100, 100, 100), mode='ct')
                total_detections += 1
            
            
            # Show Image
            cv2.imshow("Predictions", frame)
            writer.write(frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                    stop = True
                    break
        if stop:
            break

    print("Inference Complete!")

def write_label(image,x1,y1,x2,y2,label, color=(255,255,255), mode='tl'):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    label_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    if mode == 'bl':
        cv2.putText(image, label, (x1, y2 + 2*baseline), font, font_scale, color, font_thickness, cv2.LINE_AA)
    elif mode == 'ct':
        cv2.putText(image, label, (x1, y2 - baseline), font, font_scale, color, font_thickness, cv2.LINE_AA)
    else:
        cv2.putText(image, label, (x1, y1 - baseline), font, font_scale, color, font_thickness, cv2.LINE_AA)

''' Main Script'''
if __name__ == '__main__':
    args = init_parser().parse_args()
    print(args)
    main_func(args)
