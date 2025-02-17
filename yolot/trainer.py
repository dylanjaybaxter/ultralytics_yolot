# General
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
# Torch
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
# Ultralytics
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils import LOGGER, colorstr

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

#from torchviz import make_dot

# YOLOT
from yolot.val import SequenceValidator
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
        self.red_factor = cfg['red_factor']
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
        self.overwrite = cfg['overwrite']
        self.batch = cfg['batch']
        self.mixup = cfg['mixup']
        self.acc = cfg['acc']

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
            if os.path.exists(os.path.join(self.paths['run'], "weights", "last.pt")) and not self.overwrite:
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
                                                data_cap=100000, seq_len=self.sequence_len, aug=cfg["aug"], 
                                                drop=cfg["drop"], mixup=cfg['mixup'], args=cfg, batch=self.batch)

        # Build validators
        self.validator = self.build_validator(data_path=self.paths['data'], limit=100000, seq_len=6)
        self.mini_validator = self.build_validator(data_path=self.paths['data'], limit=40, seq_len=6)

        # Build Optimizer and Gradient Scaler
        self.scaler = GradScaler(enabled=True)
        # Define Optimizer and Scheduler
        self.lam1 = lambda epoch: max((self.red_factor ** epoch), self.lrf/self.lr0)
        # Reset learning rate
        if self.ckpt is not None and self.continuing:
            if 'metadata' in self.ckpt:
                self.lr0= self.lr0 * self.lam1(self.ckpt['metadata']['epoch']) 
                print(f"Continuing with learning rate {self.lr0}")
        
        #self.optimizer = opt.SGD(self.model.parameters(), lr=self.lr0, momentum=self.momentum)
        self.optimizer = self.build_optimizer(self.model, name=self.cfg['optimizer'], lr=self.lr0, momentum=self.momentum)

        if self.ckpt is not None and self.continuing:
            # Load state of previous optimizer
            self.optimizer.load_state_dict(self.ckpt['optimizer'])

        # If loading from a checkpoint, load the optimizer
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lam1)

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

    def build_dataloader(self, data_path, split, data_cap, seq_len, aug=False, 
                         drop=0.0, args=None, mixup=0, batch=1, cf=collate_fn):
        # Create Datasets for Training and Validation
        dataset = BMOTSDataset(data_path, split,
                               device=self.device,
                               seq_len=seq_len,
                               data_cap=data_cap,
                               aug=aug,
                               drop=drop,
                               mixup=mixup,
                               args=args)
        if self.ddp:
            # Create Samplers for distributed processing
            sampler = DistributedSampler(dataset, shuffle=False,
                                               drop_last=False)
            # Create Dataloader with reusable workers
            dataloader = InfiniteDataLoader(dataset, num_workers=self.workers, batch_size=batch, shuffle=False,
                                            collate_fn=cf, drop_last=False, pin_memory=False,
                                            sampler=sampler)
        else:
            # Create a dataloader with reusable workers
            sampler = None
            dataloader = InfiniteDataLoader(dataset, num_workers=self.workers, batch_size=batch, shuffle=False,
                                            collate_fn=cf, drop_last=False, pin_memory=False)
        return dataloader

    def build_validator(self, data_path, limit, seq_len):
        # Create Validator
        val_loader = self.build_dataloader(data_path=data_path, split="val", data_cap=limit, seq_len=seq_len, cf=single_batch_collate)
        validator = SequenceValidator(dataloader=val_loader, save_dir=Path(self.paths['mini']))
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
        self.model.zero_states()
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
        warmup= True
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
                self.model.train()
                # Update iteration counter
                iteration = (epoch - 1) * len(self.dataloader) + seq_idx
                # Warmup Logic
                if self.nw > warmup_counter:
                    for idx, x in enumerate(self.optimizer.param_groups):
                        x['lr'] = np.interp(warmup_counter, [0,self.nw], [0.0, self.lr0])
                        if "momentum" in x: 
                            x["momentum"] = np.interp(warmup_counter, [0,self.nw], [self.warmup_momentum, self.momentum])
                    warmup_counter += 1
                elif warmup:
                    print("Exiting Warmup...")
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
                
                # Clear gradients the iteration after accumulation
                if ((seq_idx+2 % self.acc) == 0) or (self.acc == 1):
                    # Zero Out Leftover Gradients
                    self.optimizer.zero_grad()
                # Compute New Gradients Always
                self.scaler.scale(loss/self.acc).backward()
                # Update only when accumulated acc batches
                if ((seq_idx+1 % self.acc) == 0) or (self.acc == 1):
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Update Progress Bar
                if self.global_rank == 0:
                    pbar.set_description(
                        f"Seq:{seq_idx + 1}/{num_seq}, Loss:{loss:.10e}, lr: {self.optimizer.param_groups[0]['lr']:.5e}:, W: {warmup}")
                    self.tb_writer.add_scalar('Loss', loss, iteration)
                    self.tb_writer.add_scalar('diagnostics/LR', self.optimizer.param_groups[0]['lr'], iteration)
                    self.tb_writer.add_scalar('diagnostics/im_max', subsequence['img'].max(), iteration)
                    self.tb_writer.add_scalar('diagnostics/im_std', subsequence['img'].std(), iteration)
                    pbar.refresh()

                # Save checkpoint periodically
                if self.global_rank == 0 and save_counter >= self.save_freq:
                    print("Validating...")
                    # Prevents unnecessary gradient tracking
                    with torch.no_grad():
                        if self.ddp:
                            mini_metrics = self.mini_validator(model=self.model.module)
                        else:
                            # Use validator to evaluate a mini dataset as a sanity check
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
                
                # Reset counter for mini validator
                if save_counter >= self.save_freq:
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
                                    epoch+1, 0, loss, self.optimizer.param_groups[0]['lr'],
                                    self.paths['model_save'], "last.pt")
                else:
                    self.save_checkpoint(self.model.state_dict(), self.optimizer.state_dict(),
                                         epoch+1, 0, loss, self.optimizer.param_groups[0]['lr'], 
                                         self.paths['model_save'], "last.pt")

            # Validate
            if self.global_rank == 0:
                with torch.no_grad():
                    self.model.eval()
                    metrics = self.validator(model=self.model)
                self.write_to_tb("metrics", [], metrics, epoch, all=True)
                # Save Best
                print(f"Comparing fitness({metrics['fitness']} to current best({best_metric}))...")
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
    
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        # (From ultralytics)
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.cfg.lr0}' and 'momentum={self.cfg.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer

def check_gradients_for_nan(model):
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print("NaN values detected in gradients!")
                break
            elif torch.isinf(param.grad).any():
                print("Inf values detected in gradients!")
                break
        else:
            print(f"None_Grad: {param.name} {param.requires_grad}")







