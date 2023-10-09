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
    # Setup Arguments
    model = args.model
    epochs = args.epochs
    dataset_path = args.data
    model_save_path = args.sModel
    metrics_save_path = args.sMetrics
    model_load_path = args.lModel
    visualize = args.vis
    # Overwrite any local changes with the config file
    config_path = args.conf
    if config_path:
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


    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Training on {device}")

    # Debug option
    #torch.autograd.set_detect_anomaly(True)
    scaler = GradScaler(enabled=True)

    # Initialize Parallelization
    #init_distributed()

    # Setup Dataloader
    print("Building Dataset")
    # Create Datasets for Training and Validation
    training_dataset = BMOTSDataset(dataset_path, "train", device=device, seq_len=sequence_len)
    val_dataset = BMOTSDataset(dataset_path, "val", device=device, seq_len=sequence_len)
    # Use Datasets to Create Autoloader
    train_loader = DataLoader(training_dataset, num_workers=workers, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, num_workers=workers, batch_size=1, shuffle=True, collate_fn=single_batch_collate)

    # Create Model
    #model = DetectionModel(cfg="yolov8Tn.yaml")
    model = SequenceModel(cfg=model, device=device)
    model.train()
    model.model_to(device)
    model.hidden_states_to(device)
    if model_load_path:
        model.load_state_dict(torch.load(model_load_path), strict=False)
    optimizer = opt.SGD(model.parameters(), lr=lr0)

    # Attributes bandaid
    class Args(object):
        pass
    model.args = Args()
    model.args.cls = cls_gain
    model.args.box = box_gain
    model.args.dfl = dfl_gain

    #trainer = DetectionTrainer(cfg= ROOT / "cfg/t_config.yaml")
    validator = DetectionValidator(dataloader=val_loader, save_dir=Path(metrics_save_path))
    model.zero_states()

    # Test Model
    #test_input = torch.rand(6, 3, 640, 640)
    #test_output, hidden_states = model.process_sequence(test_input)

    # Main Training Loop
    model.train()
    loss = 0 # Arbitrary Starting Loss for Display
    for epoch in range(epochs):
        # Make sure model is in training mode
        model.train()
        # Set Up Loading bar for epoch
        bar_format = f"::Epoch {epoch}/{epochs}| {{bar:30}}| {{percentage:.2f}}% | [{{elapsed}}<{{remaining}}] | {{desc}}"
        pbar_desc = f'Loss: {loss:.10e}'
        pbar = tqdm(train_loader, desc=pbar_desc, bar_format=bar_format, ascii=False)
        num_seq = len(train_loader)
        # Single Epoch Training Loop
        for seq_idx, subsequence in enumerate(pbar):
            # Forward Pass
            #with autocast():
            # Evaluate Sequence
            outputs = model(subsequence[0]['img'].to(device))
            # Compute Loss
            loss = model.sequence_loss(outputs, subsequence[0])

            # If visualize, plot outputs with imshow
            if visualize:
                display_predictions(subsequence[0], outputs, 16)

            # Compute New Gradients
            #scaler.scale(loss).backward()
            loss.backward()
            # Update weights
            #scaler.step(optimizer)
            optimizer.step()
            # Reset and detach hidden states
            model.zero_states()
            # Zero Out Leftover Gradients
            optimizer.zero_grad()

            # Update Progress Bar
            pbar.set_description(f'Seq:{seq_idx}/{num_seq}, Loss:{loss:.10e}:')
            pbar.refresh()


        # Validate
        validator(model=model)

        # Save Model
        torch.save(model.state_dict(), os.path.join(model_save_path, "yolot_test.pt"))

    # Evalutation
    print("Training Complete:)")

def display_predictions(batch, preds, num_frames):
    # For first four images in batch
    images = []
    for i in range(min(num_frames,batch['img'].size()[0])):
        image = batch['img'][i, :, :, :].cpu().transpose(0,1).transpose(1,2).numpy()
        h, w = batch['ori_shape'][i]
        image = cv2.cvtColor(cv2.resize(image, (w,h)), cv2.COLOR_RGB2BGR)
        pred = torch.cat([stride.view(1,144,-1) for stride in preds[i]], dim=2)
        filtered_pred = non_max_suppression(pred, conf_thres=0.9, max_wh=1, iou_thres=0.6, classes=[0,1,2])
        # Draw Labels on Image
        color_label = (1,0,0)
        for j in range(batch['bboxes'].size()[0]):
            if batch['frame_idx'][j] == i:
                box = batch['bboxes'][j,:]
                x1 = int((box[0] - 0.5*box[2])*w)
                y1 = int((box[1] - 0.5*box[3])*h)
                x2 = int((box[0] + 0.5*box[2])*w)
                y2 = int((box[1] + 0.5*box[3])*h)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color_label, thickness=2)

        color_pred = (0,255,0)
        for pred_ind in range(filtered_pred[0].size()[0]):
            # Check for class 1
            if filtered_pred[0][pred_ind,5] == 1.0:
                box = filtered_pred[0][pred_ind,0:4]
                conf = filtered_pred[0][pred_ind,4]
                x1 = int((box[0] - 0.5 * box[2]) * w)
                y1 = int((box[1] - 0.5 * box[3]) * h)
                x2 = int((box[0] + 0.5 * box[2]) * w)
                y2 = int((box[1] + 0.5 * box[3]) * h)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color_pred, thickness=2)

        cv2.imshow('Label Output', image)
        cv2.waitKey(1)

        # Draw Predictions on Rectangle
        beans = "beans"

    return 0

def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

''' Main Script'''
if __name__ == '__main__':
    # Setup parallel stuff
    multiprocessing.set_start_method('spawn')

    args = init_parser().parse_args()
    print(args)
    #
    if args.prof:
        cProfile.run('main_func(args)', 'outputs.prof')
        p = pstats.Stats('outputs.prof')
        p.sort_stats('cumulative').print_stats(20)
    else:
        main_func(args)
    print("Done!")

