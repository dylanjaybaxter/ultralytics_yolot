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
default_model_path = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\med_gru\\weights\\best_15e.pth"
default_base = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\test_run_big_data\\weights\\best.pt"
default_save_dir = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\yolot\\val_runs"
default_vid_name = "base_only.mp4"
default_data_path = "C:\\Users\\dylan\\Documents\\Data\\BDD100k_MOT202\\bdd100k"
default_model_conf = "cfg/models/yolot_gru_bigm.yaml"
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
    baseline = args.base


    # Build Model
    model = SequenceModel(cfg=model_cfg, device=0)
    model.eval()
    # Save Weights
    model.load_state_dict(torch.load(model_path)['model'])
    model.model_to(0)

    # Build Baseline
    base_model = YOLO(baseline, task="predict")
    base_model.to(0)

    # Set up val dataloader
    print("Building Dataset...")
    val_dataset = BMOTSDataset(data_path, "val", device=0, seq_len=100, data_cap=1000, shuffle=False)
    val_loader = InfiniteDataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False,
                            collate_fn=single_batch_collate, drop_last=False, pin_memory=False)

    # Create Validator
    #validator = SequenceValidator(val_loader, iou_thres=0.4, conf_thres=0.25, device=0, ddp=False)
    #validator = SequenceValidator2(dataloader=val_loader)
    #stats = validator(model=model)

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
    base_total_detections = 0
    model.zero_states()
    stop = False
    for seq_idx, subsequence in enumerate(pbar):

        sub_ims = subsequence['img']
        with torch.no_grad():
            outputs = model(sub_ims)

        for frame_idx in range(sub_ims.shape[0]):
            bdets = base_model(sub_ims[frame_idx,:,:,:].unsqueeze(0))

            # Preprocess Frame for OpenCV
            frame = sub_ims[frame_idx,:,:,:].cpu().numpy()
            frame = (np.transpose(frame, (1,2,0))*255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Process Detections
            raw_dets = outputs[frame_idx][0]
            # NMS
            dets = non_max_suppression(raw_dets, conf_thres=nms_conf, iou_thres=nms_iou)

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
            
            # # Display Detections
            # for det_idx in range(dets[0].shape[0]):
            #     conf = float(dets[0][det_idx,4])
            #     cls = int(dets[0][det_idx, 5])
            #     x1 = int(dets[0][det_idx,0])
            #     y1 = int(dets[0][det_idx,1])
            #     x2 = int(dets[0][det_idx,2])
            #     y2 = int(dets[0][det_idx,3])
            #     cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0), thickness=3)
            #     write_label(frame, x1, y1,x2,y2, f"{label_dict[cls]}:{conf:.2f}")
            #     total_detections += 1
            
            # Display Baseline
            base_boxes = bdets[0].boxes.xyxy.to(int)
            base_cls = bdets[0].boxes.cls.to(int)
            base_conf = bdets[0].boxes.conf.to(float) 
            for det_idx in range(base_boxes.shape[0]):
                conf = float(base_conf[det_idx])
                cls = int(base_cls[det_idx])
                x1 = int(base_boxes[det_idx][0])
                y1 = int(base_boxes[det_idx][1])
                x2 = int(base_boxes[det_idx][2])
                y2 = int(base_boxes[det_idx][3])
                cv2.rectangle(frame,(x1,y1),(x2,y2), (0,0,150), thickness=2)
                write_label(frame, x1, y1,x2,y2, f"{label_dict[cls]}:{conf:.2f}", color=(0,0,255), mode='bl')
                base_total_detections += 1

            print(f"Sequence {seq_idx}, Frame {frame_idx}: {dets[0].shape[0]} detections, {base_boxes.shape[0]} baseline")

            # Show Image
            cv2.imshow("Predictions", frame)
            writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
