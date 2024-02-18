# General
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
# Torch
import torch
import torch.optim as opt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
# Ultralytics
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.ops import non_max_suppression
# Parallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
# Logging
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import atexit
# Profiling
import cProfile
import pstats
import cv2
import numpy as np

# YOLOT
from yolot.val import SequenceValidator2
from yolot.BMOTSDataset import BMOTSDataset, collate_fn, single_batch_collate
from yolot.SequenceModel import SequenceModel

class YolotTrainer():
    def __init__(self, cfg=None):
        assert cfg is not None
        # Path and Port Config
        self.paths = {}
        self.cfg = cfg
        self.paths['model_name'] = cfg['model']
        self.paths['data'] = cfg['data']
        self.paths['tb_dir'] = cfg['log_dir']
        self.paths['base'] = cfg['met_save_path']
        self.paths['model_load'] = cfg['model_load_path']
        self.run_name = cfg['run_name']
        self.tb_port = cfg['log_port']
        # Training Setup
        self.sequence_len = cfg['seq_len']
        self.epochs = cfg['epochs']
        self.gains = {}
        self.gains['cls'] = cfg['cls']
        self.gains['box'] = cfg['box']
        self.gains['dfl'] = cfg['dfl']
        self.lr0 = cfg['lr0']
        self.lrf = cfg['lrf']
        self.momentum = cfg['momentum']
        self.warmup_momentum = cfg['warmup_momentum']
        self.nw = cfg['warmup_its']
        self.save_freq = cfg['save_freq']
        self.workers = cfg['workers']
        # Debug Settings
        self.seq_cap = cfg['seq_cap']
        self.visualize = cfg['visualize']
        self.prof = cfg['prof']
        self.ddp = cfg['ddp']
        self.DEBUG = cfg['DEBUG']

        # Setup Device
        mp.set_start_method('spawn')
        self.setup_DDP()
        self.setup_device()

        # Setup Save Directories
        self.paths['run'] = os.path.join(self.paths['base'], self.run_name)
        self.paths['mini'] = os.path.join(self.paths['run'], "mini")
        if self.global_rank == 0:
            # Create File structure for the run
            if not os.path.exists(self.paths['run']):
                os.mkdir(os.path.join(self.paths['run']))
                os.mkdir(os.path.join(self.paths['run'], "weights"))
                os.mkdir(os.path.join(self.paths['run'], "other"))
                os.mkdir(os.path.join(self.paths['run'], "mini"))
                os.mkdir(os.path.join(self.paths['run'], "tb"))
        # If run directory already exists, look for checkpoint
        if os.path.exists(self.paths['run']):
            # Look for checkpoint
            print(f"Continuing Run: {self.run_name}")
            if os.path.exists(os.path.join(self.paths['run'], "weights", "last.pt")):
                self.paths['model_load'] = os.path.join(self.paths['run'], "weights", "last.pt")
                print(f"Using previous checkpoint: {self.paths['model_load']}")
                self.continuing = True
            else:
                print(f"Starting run from scratch with weights: {self.paths['model_load']}")
                self.continuing = False
        else:
            self.continuing = False
            # Create new file structure
            print(f"Creating new run: {self.run_name}")
        self.paths['model_save'] = os.path.join(self.paths['run'], "weights")
        self.paths['tb_dir'] = os.path.join(self.paths['run'], "tb")
        print(f"Saving run to {self.paths['run']}")

        # Load Model
        self.model = self.build_model(model_conf=self.paths['model_name'], model_load=self.paths['model_load'])

        # Build Dataloader
        self.dataloader = self.build_dataloader(data_path=self.paths['data'], split="train",
                                                data_cap=100000, seq_len=self.sequence_len)

        # Build validators
        self.validator = self.build_validator(data_path=self.paths['data'], limit=100000, seq_len=6)
        self.mini_validator = self.build_validator(data_path=self.paths['data'], limit=40, seq_len=6)

        # Build Optimizer and Gradient Scaler
        self.scaler = GradScaler(enabled=True)
        # Define Optimizer and Scheduler
        self.lam1 = lambda epoch: max((0.9 ** epoch), self.lrf/self.lr0)
        self.optimizer = opt.SGD(self.model.parameters(), lr=self.lr0, momentum=self.momentum)
        if self.ckpt is not None and self.continuing:
            # Load state of previous optimizer
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            # Reset learning rate
            if self.ckpt['metadata']['epoch']:
                self.lr0= self.lr0 * self.lam1(self.ckpt['metadata']['epoch']) 
                print(f"Continuing with learning rate {self.lr0}")
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr0

        # If loading from a checkpoint, load the optimizer
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=[self.lam1])

        # Initialize Tensorboard
        self.tb_writer, self.tb_prog = self.init_tb(self.paths['tb_dir'], port=self.tb_port)

        # Setup Cleanup
        atexit.register(self.cleanup)

    def setup_device(self):
        print(torch.__version__)
        print(torch.cuda.nccl.is_available(torch.randn(1).cuda()))
        #print(torch.cuda.nccl.version())
        print(f"Training on GR: {self.global_rank}/{self.world_size}, LR: {self.local_rank}...checking in...")
        if torch.cuda.is_available():
            self.device = 'cuda:' + str(self.local_rank)
        else:
            self.device = torch.device('cpu')
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()

    def build_model(self, model_conf, model_load):
        model = SequenceModel(cfg=model_conf, device=self.device, verbose=(self.local_rank == 0))
        model.train()
        model.model_to(self.device)
        ckpt = None
        if os.path.exists(model_load):
            print(f"Loading model_load from {model_load}")
            self.ckpt = torch.load(model_load)
            if 'model' in self.ckpt:
                model.load_state_dict(self.ckpt['model'], strict=False)
            else:
                model.load_state_dict(self.ckpt)
            if 'metadata' in self.ckpt:
                print(self.ckpt['metadata'])
        else:
            self.ckpt = None
        print(f"Building parallel model_load with device: {torch.device(self.device)}")

        # Attributes bandaid
        class Args(object):
            pass
        model.args = Args()
        model.args.cls = self.gains['cls']
        model.args.box = self.gains['box']
        model.args.dfl = self.gains['dfl']

        if self.ddp:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

        return model

    def build_dataloader(self, data_path, split, data_cap, seq_len):
        # Create Datasets for Training and Validation
        dataset = BMOTSDataset(data_path, split,
                               device=self.device,
                               seq_len=seq_len,
                               data_cap=data_cap)
        # Create Samplers for distributed processing
        if self.ddp:
            sampler = DistributedSampler(dataset, shuffle=False,
                                               drop_last=False)
            dataloader = InfiniteDataLoader(dataset, num_workers=self.workers, batch_size=1, shuffle=False,
                                            collate_fn=collate_fn, drop_last=False, pin_memory=False,
                                            sampler=sampler)
        else:
            sampler = None
            dataloader = InfiniteDataLoader(dataset, num_workers=self.workers, batch_size=1, shuffle=False,
                                            collate_fn=single_batch_collate, drop_last=False, pin_memory=False)
        return dataloader

    def build_validator(self, data_path, limit, seq_len):
        # Create Validator
        val_loader = self.build_dataloader(data_path=data_path, split="val", data_cap=limit, seq_len=seq_len)
        validator = SequenceValidator2(dataloader=val_loader, save_dir=Path(self.paths['mini']))
        validator.training = True
        return validator

    def init_tb(self, path, port):
        # Initialize Tensorboard
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = path
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port), '--bind_all'])
        url = tb.launch()
        print(f"Tensorboard started listening to {log_dir} and broadcasting on {url}")
        tb_writer = SummaryWriter(log_dir=os.path.join(path, dt))
        return tb_writer, tb

    def train_model(self):
        # Main Training Loop
        self.model.train()
        best_metric = 0.
        loss = 0  # Arbitrary Starting Loss for Display
        if self.ckpt and self.continuing:
            starting_epoch = self.ckpt['metadata']['epoch']
            skipping = True
        else:
            starting_epoch = 1
            skipping = False

        # dist.barrier()
        print(f"RANK {self.global_rank} Starting training loop")
        warmup_counter = 0
        for epoch in range(starting_epoch, self.epochs + 1):
            # Make sure model is in training mode
            self.model.train()
            if self.ddp:
                self.model.module.model_to(self.device)
                self.dataloader.sampler.set_epoch(epoch)
                self.validator.dataloader.sampler.set_epoch(epoch)
            else:
                self.model.model_to(self.device)

            # Set Up Loading bar for epoch
            bar_format = f"::Epoch {epoch}/{self.epochs}| {{bar:30}}| {{percentage:.2f}}% | [{{elapsed}}<{{remaining}}] | {{desc}}"
            pbar_desc = f"Seq:.../..., Loss: {loss:.10e}, lr: {self.optimizer.param_groups[0]['lr']:.5e}"
            pbar = tqdm(self.dataloader, desc=pbar_desc, bar_format=bar_format, ascii=False, disable=(self.global_rank != 0))
            num_seq = len(self.dataloader)

            # Single Epoch Training Loop
            save_counter = 0
            for seq_idx, subsequence in enumerate(pbar):
                # Update iteration counter
                iteration = (epoch - 1) * len(self.dataloader) + seq_idx
                # Warmup Logic
                if self.nw > warmup_counter:
                    for idx, x in enumerate(self.optimizer.param_groups):
                        x['lr'] = np.interp(warmup_counter, [0,self.nw], [0.1 if idx==0 else 0.0, self.lr0])
                        if "momentum" in x:
                            x["momentum"] = np.interp(warmup_counter, [0,self.nw], [self.warmup_momentum, self.momentum])
                    warmup = True
                    warmup_counter += 1
                else:
                    warmup = False
                # Skip iterations if checkpoint
                if skipping and self.ckpt['metadata']['iteration'] > seq_idx and \
                        self.ckpt['metadata']['iteration'] < num_seq - 10:
                    pbar.set_description(
                        f"Seq:{seq_idx + 1}/{num_seq}, Skipping to idx{self.ckpt['metadata']['iteration']}:")
                    pbar.refresh()
                    continue
                else:
                    skipping = False
                # Reset and detach hidden states
                if self.ddp:
                    self.model.module.zero_states()
                else:
                    self.model.zero_states()
                # Forward Pass
                with autocast(enabled=False):
                    outputs = self.model(subsequence['img'].to(self.device))
                    # Compute Loss
                    if self.ddp:
                        loss = self.model.module.sequence_loss(outputs, subsequence)
                    else:
                        loss = self.model.sequence_loss(outputs, subsequence)
                 
                #self.display_predictions(subsequence, outputs, 1)

                # Zero Out Leftover Gradients
                self.optimizer.zero_grad()
                # Compute New Gradients
                self.scaler.scale(loss).backward()
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update Progress Bar
                if self.global_rank == 0:
                    pbar.set_description(
                        f"Seq:{seq_idx + 1}/{num_seq}, Loss:{loss:.10e}, lr: {self.optimizer.param_groups[0]['lr']:.5e}:, W: {warmup}")
                    self.tb_writer.add_scalar('Loss', loss, iteration)
                    self.tb_writer.add_scalar('diagnostics/LR', self.scheduler.get_last_lr()[0], iteration)
                    self.tb_writer.add_scalar('diagnostics/im_max', subsequence['img'].max(), iteration)
                    self.tb_writer.add_scalar('diagnostics/im_std', subsequence['img'].std(), iteration)
                    pbar.refresh()

                # Save checkpoint periodically
                if self.global_rank == 0 and save_counter > self.save_freq:
                    print("Validating...")
                    with torch.no_grad():
                        if self.ddp:
                            mini_metrics = self.mini_validator(model=self.model.module)
                        else:
                            self.model.eval()
                            mini_metrics = self.mini_validator(model=self.model, fuse=False)
                            self.model.train()
                    self.write_to_tb("mini_metrics", [], mini_metrics, iteration, all=True)

                    if self.ddp:
                        self.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                        epoch, seq_idx, loss, self.optimizer.param_groups[0]['lr'], 
                                        self.paths['run'], "mini_check.pt")
                    else:
                        self.save_checkpoint(self.model.state_dict(), self.optimizer.state_dict(),
                                             epoch, seq_idx, loss, self.optimizer.param_groups[0]['lr'], 
                                             self.paths['run'], "mini_check.pt")

                if save_counter > self.save_freq:
                    save_counter = 0
                    if self.ddp:
                        dist.barrier()
                else:
                    save_counter += 1

                # Exit early for debug
                if self.DEBUG and seq_idx >= self.seq_cap:
                    print(".................Breaking Early for debug reasons..................")
                    break

            # Save Checkpoint
            if self.global_rank == 0:
                print(f"Saving checkpoint to {os.path.join(self.paths['model_save'], 'last.pt')}")
                if self.ddp:
                    self.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                    epoch, 0, loss, self.optimizer.param_groups[0]['lr'],
                                    self.paths['model_save'], "last.pt")
                else:
                    self.save_checkpoint(self.model.state_dict(), self.optimizer.state_dict(),
                                         epoch, 0, loss, self.optimizer.param_groups[0]['lr'], 
                                         self.paths['model_save'], "last.pt")

            # Validate
            if self.global_rank == 0:
                metrics = self.validator(model=self.model)
                self.write_to_tb("metrics", [], metrics, epoch, all=True)
                # Save Best
                if metrics['fitness'] >= best_metric:
                    print(f"Saving new best to {self.paths['model_save']}")
                    if self.ddp:
                        self.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                        epoch, 0, loss, self.optimizer.param_groups[0]['lr'], 
                                        self.paths['model_save'], "best.pth")
                    else:
                        self.save_checkpoint(self.model.state_dict(), self.optimizer.state_dict(),
                                             epoch, 0, loss, self.optimizer.param_groups[0]['lr'], 
                                             self.paths['model_save'], "best.pth")
                    best_metric = metrics['fitness']

            # Detach tensors
            self.scheduler.step()
        # Cleanup
        self.cleanup()

        # Evalutation
        print("Training Complete:)")

    def write_to_tb(self, prefix, topics, data, iter, all=False):
        if all:
            keys = data.keys()
        else:
            keys = topics
        for key in keys:
            keyname = key.split("/")[-1]
            self.tb_writer.add_scalar(f"{prefix}/{keyname}", data[key], iter)

    
    def save_checkpoint(self, model_dict, opt_dict, epoch, itr, loss, lr, save_path, save_name):
        metadata = {
            'epoch': epoch,
            'iteration': itr,
            'loss': loss,
            'lr': lr,
        }
        save_obj = {
            'model': model_dict,
            'optimizer': opt_dict,
            'metadata': metadata,
        }
        torch.save(save_obj, os.path.join(save_path, save_name))

    def setup_DDP(self):
        try:
            self.init_distributed()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
        except:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1

    def init_distributed(self):
        # Initialize Parallelization
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ['OMP_NUM_THREADS'] = "2"
        torch.multiprocessing.set_sharing_strategy("file_system")
        dist.init_process_group(backend="gloo")

    def cleanup(self):
        self.tb_writer.close()
        if self.ddp:
            dist.destroy_process_group()
        #self.tb_prog.kill()
    
    def display_predictions(self, batch, preds, num_frames):
        # For first four images in batch
        images = []
        for i in range(min(num_frames,batch['img'].size()[0])):
            image = batch['img'][i, :, :, :].cpu().transpose(0,1).transpose(1,2).numpy()
            h, w = batch['ori_shape'][i]
            image = cv2.cvtColor(cv2.resize(image, (w,h)), cv2.COLOR_RGB2BGR)
            pred = torch.cat([stride.view(1,144,-1) for stride in preds[i]], dim=2)
            filtered_pred = non_max_suppression(pred, conf_thres=0.9, max_wh=1, iou_thres=0.6, classes=[0,1,2])
            # Draw Labels on Image
            color_label = (255,0,0)
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







