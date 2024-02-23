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
from yolot.val import SequenceValidator2


# Defaults and Macros
default_model_path = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\best.pth"
default_save_dir = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\yolot\\val_runs"
default_vid_path = "loss_fix"
default_data_path = "C:\\Users\\dylan\\Documents\\Data\\BDD100k_MOT202\\bdd100k"
default_device = 0
FRAME_RATE = 5.0

# Options Parser
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to model to be trained')
    parser.add_argument('--run_name', type=str, default=default_vid_path, help='Path to inference video ')
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default=default_save_dir, help="Path to which visualization will be saved")
    parser.add_argument('--data', type=str, default=default_data_path, help="Path to Dataset")
    parser.add_argument('--device', type=int, default=default_device, help="device")
    return parser

# Main Func
def main_func(args):
    # Establish args
    model_path = args.model_path
    data_path = args.data
    DISP = args.disp
    save_path = args.save_path
    device = args.device


    # Build Model
    model = SequenceModel(cfg="yolov8Tn_GRU.yaml", device=0)
    model.eval()
    # Save Weights
    model.load_state_dict(torch.load(model_path)['model'])
    model.model_to(0)

    # Set up val dataloader
    print("Building Dataset...")
    val_dataset = BMOTSDataset(data_path, "val", device=0, seq_len=24, data_cap=4)
    val_loader = InfiniteDataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False,
                            collate_fn=single_batch_collate, drop_last=False, pin_memory=False)

    sample_data = val_dataset[0]

    # Create Validator
    #validator = SequenceValidator(val_loader, iou_thres=0.4, conf_thres=0.25, device=0, ddp=False)
    #validator = SequenceValidator2(dataloader=val_loader)
    #stats = validator(model=model)

    # Setup Video Writer
    if save_path:
        writer = cv2.VideoWriter(os.path.join(save_path, "test_vid.mp4"), 
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
    model.zero_states()
    for seq_idx, subsequence in enumerate(pbar):

        sub_ims = subsequence['img']
        with torch.no_grad():
            outputs = model(sub_ims)

        for frame_idx in range(sub_ims.shape[0]):
            # Preprocess Frame for OpenCV
            frame = sub_ims[frame_idx,:,:,:].cpu().numpy()
            frame = (np.transpose(frame, (1,2,0))*255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Process Detections
            raw_dets = outputs[frame_idx][0]
            # NMS
            dets = non_max_suppression(raw_dets, conf_thres=0.6, iou_thres=0.25)
            print(f"Sequence {seq_idx}, Frame {frame_idx}: {dets[0].shape[0]} detections, {total_detections} total detections")
            for det_idx in range(dets[0].shape[0]):
                print(dets[0][det_idx, :])
                conf = dets[0][det_idx,4]
                cls = dets[0][det_idx, 5]
                x1 = int(dets[0][det_idx,0])
                y1 = int(dets[0][det_idx,1])
                x2 = int(dets[0][det_idx,2])
                y2 = int(dets[0][det_idx,3])
                #x2 = x1+w
                #y2 = y1+h
                cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0))
                total_detections += 1

            # Show Image
            cv2.imshow("Predictions", frame)
            writer.write(frame)
            cv2.waitKey(1)

    print("Inference Complete!")

''' Main Script'''
if __name__ == '__main__':
    args = init_parser().parse_args()
    print(args)
    main_func(args)
