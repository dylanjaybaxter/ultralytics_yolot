'''
This is a script intended for training YOLOT: A modified recurrent variant of YOLOv8
Author: Dylan Baxter
Created: 8/7/23
'''
import datetime

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
from ultralytics.models.yolo.detect.val import SequenceValidator
from ultralytics.cfg import ROOT
from ultralytics.nn.SequenceModel import SequenceModel
import torch.optim as opt
from tqdm import tqdm
from ultralytics.utils.ops import non_max_suppression
from torchvision.transforms.functional import resize
from torch.cuda.amp import autocast, GradScaler

# Parallelization
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from ultralytics.data.build import InfiniteDataLoader
from torch.optim.lr_scheduler import LambdaLR

# Logging
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

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

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ['WORLD_SIZE'])

'''Function to add arguments'''
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
    print ("Hello Training Weee2")
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
        #model_save_path = conf['model_save_path']
        metrics_save_path = conf['met_save_path']
        #model_load_path = conf['pt_load_path']
        visualize = conf['visualize']
        sequence_len = conf['seq_len']
        cls_gain = conf['cls']
        box_gain = conf['box']
        dfl_gain = conf['dfl']
        lr0 = conf['lr0']
        DEBUG = conf['DEBUG']
        prof = conf['prof']
        log_dir = conf['log_dir']
        log_port = conf['log_port']
        run_name = conf['run_name']
        seq_cap = conf['seq_cap']
        save_freq = conf['save_freq']


    if global_rank == 0:
        # Create File structure for the run
        # Read list of existing runs
        dirs = os.listdir(metrics_save_path)
        # If run directory already exists, look for checkpoint
        if os.path.exists(os.path.join(metrics_save_path, run_name)):
            # Look for checkpoint
            print(f"Continuing Run: {run_name}")
            if os.path.exists(os.path.join(metrics_save_path, run_name, "weights", "checkpoint.pth")):
                model_load_path = os.path.join(metrics_save_path, run_name, "weights", "checkpoint.pth")
                print("Using previous checkpoint...")
            else:
                print("Starting model from scratch")
            model_save_path = os.path.join(metrics_save_path, run_name, "weights")
            model_save_name = "checkpoint.pth"
            log_dir = os.path.join(metrics_save_path, run_name, "tb")
        else:
            # Create new file structure
            print(f"Creating new run: {run_name}")
            os.mkdir(os.path.join(metrics_save_path, run_name))
            os.mkdir(os.path.join(metrics_save_path, run_name, "weights"))
            model_load_path = ""
            model_save_path = os.path.join(metrics_save_path, run_name, "weights")
            model_save_name = "checkpoint.pth"
            os.mkdir(os.path.join(metrics_save_path, run_name, "tb"))
            log_dir = os.path.join(metrics_save_path, run_name, "tb")
            os.mkdir(os.path.join(metrics_save_path, run_name, "other"))

        # Initialize Tensorboard
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, dt)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', str(log_port), '--bind_all'])
        url = tb.launch()
        print(f"Tensorboard started listening to {log_dir} and broadcasting on {url}")
        tb_writer = SummaryWriter(log_dir=log_dir)



    # Setup Device
    print_cuda_info()
    print(f"Training on GR: {global_rank}/{world_size}, LR: {local_rank}...checking in...")
    if torch.cuda.is_available():
        device = 'cuda:' + str(local_rank)
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    # Debug option
    torch.autograd.set_detect_anomaly(True)
    scaler = GradScaler(enabled=True)

    # Setup Dataloader
    if global_rank == 0:
        print("Building Dataset")
    # Create Datasets for Training and Validation
    training_dataset = BMOTSDataset(dataset_path, "train", device=device, seq_len=sequence_len)
    val_dataset = BMOTSDataset(dataset_path, "val", device=device, seq_len=sequence_len)
    # Create Samplers for distributed processing
    train_sampler = DistributedSampler(training_dataset, shuffle=False,
                                       drop_last=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False,
                                       drop_last=False)
    # Use Datasets to Create Autoloader
    train_loader = InfiniteDataLoader(training_dataset, num_workers=workers, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, drop_last=False, pin_memory=False, sampler=train_sampler)
    val_loader = InfiniteDataLoader(val_dataset, num_workers=workers, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, drop_last=False, pin_memory=False, sampler=val_sampler)

    # Initialize Model
    model = SequenceModel(cfg=model, device=device, verbose=(local_rank==0))
    model.train()
    model.model_to(device)
    ckpt = None
    if os.path.exists(model_load_path):
        print(f"Loading model from {model_load_path}")
        ckpt = torch.load(model_load_path)
        model.load_state_dict(ckpt['model'], strict=False)

    print(f"Building parallel model with device: {torch.device(device)}")
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    print("model built")

    # Define Optimizer and Scheduler
    optimizer = opt.SGD(model.parameters(), lr=lr0, momentum=0.9)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    lam1 = lambda epoch: (0.9 ** epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=[lam1])

    # Attributes bandaid
    class Args(object):
        pass
    model.args = Args()
    model.args.cls = cls_gain
    model.args.box = box_gain
    model.args.dfl = dfl_gain

    # Create Validator and make sure that model states are zeroed
    validator = SequenceValidator(dataloader=val_loader, device='cpu')
    validator.dataloader.sampler.set_epoch(0)
    model.eval()
    model.module.zero_states()
    #validator.validate(model)

    # Main Training Loop
    model.train()
    best_state = model.module.state_dict()
    best_metric = 0
    loss = 0 # Arbitrary Starting Loss for Display
    if ckpt:
        starting_epoch = ckpt['metadata']['epoch']
        skipping = True
    else:
        starting_epoch = 1
        skipping = False


    for epoch in range(starting_epoch,epochs+1):
        # Make sure model is in training mode
        model.train()
        model.module.model_to(device)
        train_loader.sampler.set_epoch(epoch)
        validator.dataloader.sampler.set_epoch(epoch)

        # Set Up Loading bar for epoch
        bar_format = f"::Epoch {epoch}/{epochs}| {{bar:30}}| {{percentage:.2f}}% | [{{elapsed}}<{{remaining}}] | {{desc}}"
        pbar_desc = f"Seq:.../..., Loss: {loss:.10e}, lr: {optimizer.param_groups[0]['lr']:.5e}"
        pbar = tqdm(train_loader, desc=pbar_desc, bar_format=bar_format, ascii=False, disable=(global_rank != 0))
        num_seq = len(train_loader)

        # Single Epoch Training Loop
        save_counter = 0
        for seq_idx, subsequence in enumerate(pbar):
            # Skip iterations if checkpoint
            if ckpt and ckpt['metadata']['iteration'] > seq_idx and skipping and ckpt['metadata']['iteration'] < num_seq-10:
                pbar.set_description(
                    f"Seq:{seq_idx + 1}/{num_seq}, Skipping to idx{ckpt['metadata']['iteration']}:")
                pbar.refresh()
                continue
            else:
                skipping = False
            # Reset and detach hidden states
            model.module.zero_states()
            # Forward Pass
            with autocast(enabled=True):
                outputs = model(subsequence[0]['img'].to(device))
                # Compute Loss
                loss = model.module.sequence_loss(outputs, subsequence[0])

            # If visualize, plot outputs with imshow
            if visualize:
                display_predictions(subsequence[0], outputs, 16)

            # Zero Out Leftover Gradients
            optimizer.zero_grad()
            # Compute New Gradients
            scaler.scale(loss).backward()
            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Update Progress Bar
            if global_rank == 0:
                pbar.set_description(f"Seq:{seq_idx+1}/{num_seq}, Loss:{loss:.10e}, lr: {optimizer.param_groups[0]['lr']:.5e}:")
                tb_writer.add_scalar('Loss', loss, (epoch-1)*len(train_loader)+seq_idx)
                pbar.refresh()

            # Save checkpoint periodically
            if global_rank == 0 and save_counter > save_freq:
                save_checkpoint(model.module.state_dict(), optimizer.state_dict(),
                                epoch, seq_idx, loss, model_save_path, model_save_name)

            # Exit early for debug
            if DEBUG and seq_idx >= seq_cap:
                break

        # Save Checkpoint
        if global_rank == 0:
            print(f"Saving checkpoint to {os.path.join(model_save_path, model_save_name)}")
            save_checkpoint(model.module.state_dict(), optimizer.state_dict(),
                            epoch, 0, loss, model_save_path, model_save_name)

        # Validate
        model.eval()
        metrics = validator.validate(model)
        if global_rank == 0:
            tb_writer.add_scalar('mAP_50',metrics['map_50'], epoch)
            #tb_writer.add_scalar('mAR', metrics['mar_100'], epoch)

        # Save Best
        if metrics['map_50'] >= best_metric:
            print(f"Saving new best to {model_save_path}")
            save_checkpoint(model.module.state_dict(), optimizer.state_dict(),
                            epoch, 0, loss, model_save_path, "best.pth")

        # Detach tensors
        scheduler.step()

    # Cleanup
    tb_writer.close()
    dist.destroy_process_group()
    tb.kill()
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
    # Initialize Parallelization
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['OMP_NUM_THREADS'] = "2"
    dist.init_process_group(backend="gloo")

def print_cuda_info():
    print(torch.__version__)
    print(torch.cuda.nccl.is_available(torch.randn(1).cuda()))
    print(torch.cuda.nccl.version())

def save_checkpoint(model_dict, opt_dict, epoch, itr, loss, save_path, save_name):
    metadata = {
        'epoch': epoch,
        'iteration': itr,
        'loss': loss,
    }
    save_obj = {
        'model': model_dict,
        'optimizer': opt_dict,
        'metadata': metadata,
    }
    torch.save(save_obj, os.path.join(save_path, save_name))


''' Main Script'''
if __name__ == '__main__':
    init_distributed()
    args = init_parser().parse_args()
    print(args)
    main_func(args)
    '''
    # Setup parallel stuff
    mp.set_start_method('spawn')
    world_size = 3
    print(f"Starting training with {world_size} workers")
    mp.spawn(
        main_func,
        args=(world_size, args),
        nprocs=world_size
    )
    dist.destroy_process_group()
    '''

    print("Done!")

