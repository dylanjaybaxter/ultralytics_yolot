# Imports
import json

import torch
from torch.utils.data import Dataset, DataLoader
import os
from os import path
from datetime import datetime
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import random
from torchvision.io import read_image
from ultralytics.data.augment import RandomHSV, RandomPerspective, RandomFlip, Compose
import numpy as np
import cv2

# BDD100k Classes
class_dict = {
    "pedestrian": 0,
    "other person": 0,
    "rider": 1,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "other vehicle": 2,
    "trailer": 6
}
label_dict = {
    0:"pedestrian",
    1:"bicycle",
    2:"car",
    3:"motorcycle",
    5:"bus",
    6:"train",
    7:"truck",
}

# Dataset object definition
class BMOTSDataset(Dataset):
    def __init__(self, base_dir, split, seq_len=16, 
                 input_size=[3,640,640], device='cpu', data_cap=None, shuffle=True, 
                 border=0, aug=False, drop=0.0, mixup=0, args=None):
        '''
        Initializes dataset by storing paths to images and labels
        :param base_dir: The base directory of the dataset
        :param transform: Transformations to be performed on data

        :attributes
        video_paths - paths to video directories
        label_paths - paths to label files
        nv - number of videos
        '''
        self.base_dir = base_dir
        self.input_size = input_size
        self.resz = Resize(input_size[1:], antialias=True)
        self.max_sequence_length = seq_len
        self.device = device
        self.border = border
        self.aug = aug
        self.drop = drop
        self.mixup = int(mixup)
        
        # Setup Augmentation
        if args == None:
            self.perspective = RandomPerspective(degrees=0.3, translate=0.3, scale=0.5, shear=0.3, perspective=0.001)
            self.perspective.size = (int(input_size[1]+border), int(input_size[2]+border))
            self.hgain = 0.015
            self.sgain = 0.7
            self.vgain = 0.4
        else:
            self.perspective = RandomPerspective(degrees=args['degrees'], translate=args['translate'], 
                                                 scale=args['scale'], shear=args['shear'], perspective=args['perspective'])
            self.perspective.size = (int(input_size[1]+border), int(input_size[2]+border))
            self.hgain = args['hsv_h']
            self.sgain = args['hsv_s']
            self.vgain = args['hsv_v']

        # Find and store video paths and subsequence indicies along with labels
        self.video_dir = path.join(base_dir, "images", "track", split)
        self.video_paths = []
        self.subsequence_keys = []
        # Find and store label paths
        self.label_dir = path.join(base_dir, "labels", "box_track_20", split)
        label_files = os.listdir(self.label_dir)
        self.label_paths = []

        self.num_vids = 0
        self.num_sequences = 0
        # Look through videos first, then approach labels
        for root, dirs, files in os.walk(path.join(self.video_dir)):
            for dir in dirs:
                # If the video has a matching label file...
                if dir+".json" in label_files:
                    # Determine full video path
                    video_path = path.join(self.video_dir, dir)
                    # Determine the number of frames in the video
                    frames = os.listdir(path.join(video_path))
                    num_frames = len(frames)
                    # Determine subsequence properties (ID, start frame, end frame)
                    start_frame = 0
                    end_frame = min(self.max_sequence_length-1, num_frames-1)
                    while start_frame < num_frames:
                        self.subsequence_keys.append((dir, start_frame, end_frame))
                        start_frame = start_frame + self.max_sequence_length
                        end_frame = min(end_frame + self.max_sequence_length, num_frames-1)
                        self.num_sequences = self.num_sequences + 1
                        if data_cap is not None:
                            if self.num_sequences >= data_cap:
                                break
                    self.num_vids = self.num_vids + 1
                    if data_cap is not None:
                        if self.num_sequences >= data_cap:
                            break
                else:
                    print(f"No label for video of ID {dir}")
        if shuffle:
            random.seed(1)
            random.shuffle(self.subsequence_keys)
        print(f"Dataset constructed with {self.num_vids} videos, creating {self.num_sequences} "
              f"sequences of max size {self.max_sequence_length}")

    def __len__(self):
        '''
        Function returns information about the size of the dataset
        :return:
        - number of videos in dataset
        '''
        return len(self.subsequence_keys)

    def __getitem__(self, idx, suppress_mixup=False):
        '''
        Function for returning a pytorch tensor of
        :param idx:
        :return:
        input frames as a tensor stack, output as tensor stack
        output:
        '''
        # Create Tensor Stack of Input Images
        seq_id, start_frame, end_frame = self.subsequence_keys[idx]
        im_paths = []
        frames = []
        ori_sizes = []
        resized_shapes = []
        bboxes = []
        cls_ids = []
        frame_ids = []
        ratio_pads = []

        # Get frames
        video_path = path.join(self.video_dir, seq_id)
        frame_files = sorted(os.listdir(video_path))
        frame_files = frame_files[start_frame:end_frame+1]
        for filename in frame_files:
            if filename.endswith(".jpg"):
                im_paths.append(filename)
                im = read_image(path.join(video_path, filename))/255.0

                # Store original size
                ori_sizes.append(tuple(im.shape[1:]))
                # Resize
                im = self.resz(im)
                # Store new size
                resized_shapes.append(tuple(im.shape[1:]))
                ratio_pads.append(
                    ((resized_shapes[-1][0] / ori_sizes[-1][0], resized_shapes[-1][1] / ori_sizes[-1][1]),
                    (0,0))) # No Padding
                frames.append(im)

        # Create Tensor Stack of Output targets
        label_path = path.join(self.label_dir, seq_id+".json")

        # Apply Augmentations
        if self.aug:
            seed = datetime.now().timestamp()
            tframes = []
            tfs = []
            scales = []
            for i, frame in enumerate(frames):
                # Set Seed
                random.seed(seed)
                # Transform Images
                # HSV
                hsv_frame = self.augment_hsv(frame, seed)
                # Affine
                tim, m, s = self.perspective.affine_transform(hsv_frame,(self.border, self.border))
                # Append results
                tfs.append(m)
                scales.append(s)
                frames[i] = torch.Tensor(tim/255.0).movedim(-1,0)
        
        # Apply Random frame dropout after halfway through the clip
        for i in range(int(len(frames)/2), len(frames)):
            if random.random() < self.drop:
                frames[i] = torch.ones(self.input_size)*0.5

        # Read in label json
        with open(label_path, 'r') as label_file:
            label_data = json.load(label_file)
        # For each frame contained
        bboxes = []
        cls_ids = []
        frame_ids = []
        # Iterate through label data
        for frame_id, frame_data in enumerate(label_data[start_frame:end_frame+1]):
            im_w = ori_sizes[frame_id][1]
            im_h = ori_sizes[frame_id][0]
            # Extract object detection data
            # Get Bounding Box Data
            for detection in frame_data["labels"]:
                # Get label ID for detection categorey
                cat = detection["category"]
                if cat not in class_dict:
                    print("No ID for: " + cat + "in " + frame_data["name"] + ", assigning to \'other vehicle\'")
                    cat_id = class_dict["other vehicle"]
                else:
                    cat_id = class_dict[cat]
                # Retrieve detection box information and calculate box center+dimensions
                x1 = (detection["box2d"]["x1"]/im_w*self.input_size[1])
                x2 = (detection["box2d"]["x2"]/im_w*self.input_size[1])
                y1 = (detection["box2d"]["y1"]/im_h*self.input_size[2])
                y2 = (detection["box2d"]["y2"]/im_h*self.input_size[2])
                box_xyxy = np.array([x1,y1,x2,y2]).astype(float)
                if self.aug:
                    box_xyxy = torch.tensor(self.perspective.apply_bboxes(np.expand_dims(box_xyxy, 0), tfs[frame_id])[0])
                else:
                    box_xyxy = torch.tensor(box_xyxy)
                box_xyxy.clamp_(0,self.input_size[2])

                w = abs(box_xyxy[2] - box_xyxy[0])
                h = abs(box_xyxy[3] - box_xyxy[1])
                cx = box_xyxy[0] + abs(box_xyxy[2] - box_xyxy[0]) / 2
                cy = box_xyxy[1] + abs(box_xyxy[3] - box_xyxy[1]) / 2

                # Append extracted data to a list
                box = torch.tensor([cx/self.input_size[1],cy/self.input_size[2],w/self.input_size[1],h/self.input_size[2]])
                if not (w==0 or h==0):
                    bboxes.append(box.clamp_(0,1))
                    cls_ids.append(torch.tensor(cat_id))
                    frame_ids.append(torch.tensor(frame_id))

        # Get Bounding Boxes
        #tboxes = torch.Tensor(self.perspective.apply_bboxes(torch.stack(bboxes, dim=0).numpy()*640, m)).to(self.device) / 640
        # Pad frames for sequence length with duplicates of the previous sequence
        if len(frame_files) < self.max_sequence_length:
            for i in range(self.max_sequence_length - len(frame_files)):
                ori_sizes.append(ori_sizes[-1])
                resized_shapes.append(resized_shapes[-1])
                ratio_pads.append(ratio_pads[-1])
                frames.append(torch.ones_like(frames[-1])*0.5)
                # frames.append(frames[-1])
                # if len(frame_ids) > 0:
                #     bboxes.append(bboxes[frame_ids==frame_ids[-1]])
                #     cls_ids.append(cls_ids[frame_ids==frame_ids[-1]])
                #     frame_ids.append(frame_ids[frame_ids==frame_ids[-1]])


        sample = {}
        sample['im_file'] = tuple(im_paths)
        sample['ori_shape'] = tuple(ori_sizes)
        sample['resized_shape'] = tuple(resized_shapes)
        sample['img'] = torch.stack(frames, dim=0).to(self.device)
        if cls_ids:
            sample['cls'] = torch.stack(cls_ids, dim=0).view(-1, 1).to(self.device)
            sample['bboxes'] = torch.Tensor(torch.stack(bboxes, dim=0)).to(self.device)
            sample['frame_idx'] = torch.stack(frame_ids).to(self.device)
        else:
            sample['cls'] = torch.Tensor([]).to(self.device)
            sample['bboxes'] = torch.Tensor([]).to(self.device)
            sample['frame_idx'] = torch.Tensor([]).to(self.device)
        sample['batch_idx'] = sample['frame_idx']  # For access by loss calculation
        sample['ratio_pad'] = ratio_pads

        # Mixup
        if not suppress_mixup:
            mixup_ratio = np.random.beta(32.0, 32.0)
            for i in range(self.mixup):
                sample2 = self.__getitem__(int(random.random()*self.__len__()), suppress_mixup=True)
                sample['img'] = mixup_ratio*sample["img"]+(1-mixup_ratio)*sample2["img"]
                sample['cls'] = torch.concat([sample['cls'], sample2['cls']])
                sample['bboxes'] = torch.concat([sample['bboxes'], sample2['bboxes']])
                sample['batch_idx'] = torch.concat([sample['batch_idx'], sample2['batch_idx']])

        return sample
    
    def augment_hsv(self, img, seed=0):
        '''Modified from ultralytics.data.augment.py RandomHSV'''
        img_np = (img.movedim(0,-1).numpy()*255).astype(np.uint8)
        np.random.seed(int(seed))
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV))
        dtype = img_np.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img_np = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB) 
        return img_np

def collate_fn(batch):
    # Merge Image Tensors for Single Forward Pass (batch, ch, w, h)
    cbatch = {}
    device = batch[0]['img'].device
    batch_ims = torch.Tensor().to(device)
    for seq in batch:
        batch_ims = torch.cat([batch_ims, seq.pop('img').unsqueeze(1)], dim=1)
    cbatch['img'] = batch_ims

    # Extend all other fields batch idx depending on frame_idx of the batch
    frame_idx_start = 0
    im_file = []
    ori_shape = []
    resized_shape = []
    bboxes = torch.Tensor().to(device)
    cls = torch.Tensor().to(device)
    frame_idx = torch.Tensor().to(device)
    for i, seq in enumerate(batch):
        im_file = im_file + list(seq['im_file'])
        ori_shape = ori_shape + list(seq['ori_shape'])
        resized_shape += list(seq["resized_shape"])
        bboxes = torch.cat([bboxes, seq['bboxes']], dim=0)
        cls = torch.cat([cls, seq['cls']], dim=0)
        frame_idx = torch.cat([frame_idx, seq['batch_idx']+frame_idx_start])
        frame_idx_start += len(im_file) 

    cbatch['im_file'] = tuple(im_file)
    cbatch['ori_shape'] = tuple(ori_shape)
    cbatch['resized_shape'] = tuple(resized_shape)
    cbatch['bboxes'] = bboxes
    cbatch['cls'] = cls
    cbatch['batch_idx'] = frame_idx
        
    return cbatch

def single_batch_collate(batch):
    return batch[0]


