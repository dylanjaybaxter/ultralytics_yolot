'''
This is a script intended for training YOLOT: A modified recurrent variant of YOLOv8
Author: Dylan Baxter
Created: 8/7/23
'''
import cv2
import torch
from torch.cuda import amp
import os
import yaml
import multiprocessing

''' Imports '''
# Standard Library
import argparse
from pathlib import Path
# Package Imports
# Local Imports
from ultralytics.data.BMOTSDataset import BMOTSDataset, collate_fn, single_batch_collate
from ultralytics.nn.tasks import parse_model, yaml_model_load
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.cfg import ROOT
from ultralytics.nn.SequenceModel import SequenceModel
import torch.optim as opt
from tqdm import tqdm
from ultralytics.utils.ops import non_max_suppression
from torchvision.transforms.functional import resize
from torch.cuda.amp import autocast, GradScaler

# Parallelization
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Profiling
import cProfile
import pstats


''' Arguments '''
# Defaults
default_dataset_path = "C:\\Users\\dylan\\Documents\\Data\\BDD100k_MOT202\\bdd100k"
default_model_path = "./model.pt"
default_num_workers = 4
DEBUG = False
default_metric_path = "C:\\Users\\dylan\\Documents\\Data\\yolot_training_results"
default_model_save_path = "C:\\Users\\dylan\\Documents\\Data\\Models\\yolot_training"
default_model_load_path = "C:\\Users\\dylan\\Documents\\Data\\Models\\yolot_training\\yolot_test.pt"
default_conf_path = "yolot_config.yaml"

''' Function to add arguments'''
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=default_model_path, help='Path to model to be trained')
    parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs to be trained')
    parser.add_argument('--data', type=str, default=default_dataset_path, help='Number of Epochs to be trained')
    parser.add_argument('--sMetrics', type=str, default=default_metric_path, help='Path to save Validation Results')
    parser.add_argument('--sModel', type=str, default=default_model_save_path, help='Path to save Final Model')
    parser.add_argument('--lModel', type=str, default=default_model_load_path, help='Path to load model from')
    parser.add_argument('--vis', type=bool, default=True, help='Show training results')
    parser.add_argument('--conf', type=str, default=default_conf_path, help="Path to configuration file")
    parser.add_argument('--enConf', type=bool, default=True, help="Enable/Disable Configuration from yaml config file")
    parser.add_argument('--prof', type=bool, default=False, help="Enable/Disable Profiling")
    return parser

''' Main Function '''
def main_func(args):
    config_path = args.conf
    # Setup Arguments
    with open(config_path, 'r') as conf_file:
        conf = yaml.safe_load(conf_file)
    model = conf['model']
    epochs = conf['epochs']
    dataset_path = conf['data']
    workers = conf['workers']
    model_save_path = conf['model_save_path']
    metrics_save_path = conf['met_save_path']
    model_load_path = conf['pt_load_path']
    visualize = conf['visualize']
    sequence_len = conf['seq_len']
    cls_gain = conf['cls']
    box_gain = conf['box']
    dfl_gain = conf['dfl']
    lr0 = conf['lr0']
    DEBUG = conf['DEBUG']

    DetectionTrainer(cfg=config_path)

    # Evalutation
    print("Training Complete:)")


''' Main Script'''
if __name__ == '__main__':
    args = init_parser().parse_args()
    main_func(args)
    print("Done!")

