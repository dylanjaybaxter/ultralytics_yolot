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

# Defaults and Macros
default_model_path = ""
default_save_dir = ""
default_vid_path = ""
FRAME_RATE = 30

# Options Parser
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to model to be trained')
    parser.add_argument('--vid_path', type=str, default=default_vid_path, help='Path to inference video ')
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default=default_save_dir, help="Path to which visualization will be saved")
    return parser

# Main Func
def main_func(args):
    # Establish args
    model_path = args.model_path
    data_path = args.data_path
    DISP = args.disp
    save_path = args.save_path

    # Load in model

    # Set Model Device and initialize

    # Read in frame paths
    if path.isdir(data_path):
        img_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        img_files.sort()

    # Determine Frame Size With Single Frame
    im = cv2.imread(img_files[0])

    # Setup Video Writer
    if save_path:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, im.size)

    # Compile
    for im_path in img_files:

        # Read in frame and convert to pytorch tensor
        im = ToTensor(cv2.imread(im_path))

        # Run prediction



    print("Inference Complete!")

''' Main Script'''
if __name__ == '__main__':
    args = init_parser().parse_args()
    print(args)
    main_func(args)
