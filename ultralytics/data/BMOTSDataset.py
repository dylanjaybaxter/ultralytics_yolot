# Imports
import json

import torch
from torch.utils.data import Dataset, DataLoader
import os
from os import path
from PIL import Image
from torchvision.transforms import ToTensor, Resize

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

# Dataset object definition
class BMOTSDataset(Dataset):
    def __init__(self, base_dir, split, seq_len=16, transform=None, input_size=[3,640,640], device='cpu'):
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
        self.transform = transform
        self.input_size = input_size
        self.resz = Resize(input_size[1:])
        self.max_sequence_length = seq_len
        self.device = device

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
                    self.num_vids = self.num_vids + 1
                else:
                    print(f"No label for video of ID {dir}")

        print(f"Dataset constructed with {self.num_vids} videos, creating {self.num_sequences} "
              f"sequences of max size {self.max_sequence_length}")



    def __len__(self):
        '''
        Function returns information about the size of the dataset
        :return:
        - number of videos in dataset
        '''
        return len(self.subsequence_keys)

    def __getitem__(self, idx):
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
        ttensor = ToTensor()

        # Get frames
        video_path = path.join(self.video_dir, seq_id)
        frame_files = sorted(os.listdir(video_path))
        frame_files = frame_files[start_frame:end_frame+1]
        for filename in frame_files:
            if filename.endswith(".jpg"):
                im_paths.append(filename)
                im = ttensor(Image.open(path.join(video_path, filename)))
                # Apply Transforms
                if self.transform:
                    im = self.transform(im)

                # Store original size
                ori_sizes.append(tuple(im.shape[1:]))
                # Resize
                im = self.resz(im, antialias=True)
                # Store new size
                resized_shapes.append(tuple(im.shape[1:]))
                ratio_pads.append((resized_shapes[-1][0] / ori_sizes[-1][0],
                                      resized_shapes[-1][1] / ori_sizes[-1][1]))
                frames.append(im)

        # Create Tensor Stack of Output targets
        label_path = path.join(self.label_dir, seq_id+".json")

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
                x1 = int(detection["box2d"]["x1"])
                x2 = int(detection["box2d"]["x2"])
                y1 = int(detection["box2d"]["y1"])
                y2 = int(detection["box2d"]["y2"])
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                cx = x1 + abs(x1 - x2) / 2
                cy = y1 + abs(y1 - y2) / 2


                # Append extracted data to a list
                cls_ids.append(torch.tensor(cat_id))
                if not all( i <= 1 for i in  [cx/im_w,cy/im_h,w/im_w,h/im_h]):
                    print("Bad value detected")

                bboxes.append(torch.tensor([cx/im_w,cy/im_h,w/im_w,h/im_h]))
                frame_ids.append(torch.tensor(frame_id))

        sample = {}
        sample['im_file'] = tuple(im_paths)
        sample['ori_shape'] = tuple(ori_sizes)
        sample['resized_shape'] = tuple(resized_shapes)
        sample['img'] = torch.stack(frames, dim=0).to(self.device)
        if cls_ids:
            sample['cls'] = torch.stack(cls_ids, dim=0).view(-1, 1).to(self.device)
            sample['bboxes'] = torch.stack(bboxes, dim=0).to(self.device)
            sample['frame_idx'] = torch.stack(frame_ids).to(self.device)
        else:
            sample['cls'] = torch.Tensor([])
            sample['bboxes'] = torch.Tensor([])
            sample['frame_idx'] = torch.Tensor([])
        sample['batch_idx'] = sample['frame_idx']  # For access by loss calculation
        sample['ratio_pad'] = ratio_pads

        return sample

def collate_fn(batch):
    return batch

def single_batch_collate(batch):
    return batch[0]
