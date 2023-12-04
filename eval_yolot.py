'''
    File: test_yolot
    Author: Dylan Baxter
    Desc: This script is intended to be used to run trained yolot models and inspect output
    Date Authored: 9/13/23
'''

# Imports
import torch
import cv2
import argparse
import os
from os import path
from torchvision.transforms import ToTensor
from tqdm import tqdm

from ultralytics.data.build import InfiniteDataLoader
from ultralytics.nn.SequenceModel import SequenceModel
from ultralytics.data.BMOTSDataset import BMOTSDataset, collate_fn


# Defaults and Macros
default_model_path = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\yolot\\good_1epoch.pth"
default_save_dir = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results\\yolot\\val_runs"
default_vid_path = "val_test"
default_data_path = "C:\\Users\\dylan\\Documents\\Data\\BDD100k_MOT202\\bdd100k"
default_device = 0
FRAME_RATE = 30

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
    model = SequenceModel(cfg="yolo8Tn.yaml", device=0)
    # Save Weights
    model.load_state_dict(torch.load(model_path).state_dict())

    # Set up val dataloader
    val_dataset = BMOTSDataset(data_path, "val", device=device, seq_len=24)
    val_loader = InfiniteDataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, drop_last=False, pin_memory=False)

    sample_data = val_dataset[0]

    # Setup Video Writer
    if save_path:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, im.size)

    # Compile
    acc = 0
    running_acc = 0
    bar_format = f"::Validation | {{bar:30}}| {{percentage:.2f}}% | [{{elapsed}}<{{remaining}}] | {{desc}}"
    pbar_desc = f"Seq:.../..., Loss: {acc:.10e}, lr:{running_acc:.5e}"
    pbar = tqdm(val_loader, desc=pbar_desc, bar_format=bar_format, ascii=False)
    num_seq = len(val_loader)

    # Single Epoch Training Loop
    save_counter = 0
    for seq_idx, subsequence in enumerate(pbar):




    print("Inference Complete!")

''' Main Script'''
if __name__ == '__main__':
    args = init_parser().parse_args()
    print(args)
    main_func(args)
