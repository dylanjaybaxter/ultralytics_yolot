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
from ultralytics.data.BMOTSDataset import BMOTSDataset, collate_fn, single_batch_collate
from ultralytics.models.yolo.detect.val import SequenceValidator, SequenceValidator2
from ultralytics.nn.SequenceModel import SequenceModel
from ultralytics.data.build import InfiniteDataLoader
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


class YolotTrainer():
    def __init__(self, cfg=None):
        assert cfg is not None
        # Path and Port Config
        self.paths = {}
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
            if os.path.exists(os.path.join(self.paths['run'], "weights", "checkpoint.pth")):
                self.paths['model_load'] = os.path.join(self.paths['run'], "weights", "checkpoint.pth")
                print("Using previous checkpoint...")
                self.continuing = True
            else:
                print("Starting model from scratch")
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
        self.optimizer = opt.SGD(self.model.parameters(), lr=self.lr0, momentum=0.9)
        # If loading from a checkpoint, load the optimizer
        if self.ckpt is not None and self.continuing:
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr0
        lam1 = lambda epoch: (0.9 ** epoch)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lam1])

        # Initialize Tensorboard
        self.tb_writer, self.tb_prog = self.init_tb(self.paths['tb_dir'], port=self.tb_port)

        # Setup Cleanup
        atexit.register(self.cleanup)

    def setup_device(self):
        print(torch.__version__)
        print(torch.cuda.nccl.is_available(torch.randn(1).cuda()))
        print(torch.cuda.nccl.version())
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
        if os.path.exists(model_load) and self.continuing:
            print(f"Loading model_load from {model_load}")
            self.ckpt = torch.load(model_load)
            model.load_state_dict(self.ckpt['model_load'], strict=False)
        elif os.path.exists(model_load) and not self.continuing:
            print(f"Loading model_load from {model_load}")
            self.ckpt = torch.load(model_load)
            model.load_state_dict(self.ckpt)
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
            '''dataloader = DataLoader(dataset, num_workers=self.workers, batch_size=1, shuffle=False,
                                            collate_fn=single_batch_collate, drop_last=False, pin_memory=False)'''
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
        log_dir = os.path.join(path, dt)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port), '--bind_all'])
        url = tb.launch()
        print(f"Tensorboard started listening to {log_dir} and broadcasting on {url}")
        tb_writer = SummaryWriter(log_dir=log_dir)
        return tb_writer, tb

    def train_model(self):
        # Main Training Loop
        self.model.train()
        best_metric = 100000000
        loss = 0  # Arbitrary Starting Loss for Display
        if self.ckpt and self.continuing:
            starting_epoch = self.ckpt['metadata']['epoch']
            skipping = True
        else:
            starting_epoch = 1
            skipping = False

        # dist.barrier()
        print(f"RANK {self.global_rank} Starting training loop")
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
                # Skip iterations if checkpoint
                if self.ckpt and self.continuing and self.ckpt['metadata']['iteration'] > seq_idx and \
                        skipping and self.ckpt['metadata']['iteration'] < num_seq - 10:
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
                        f"Seq:{seq_idx + 1}/{num_seq}, Loss:{loss:.10e}, lr: {self.optimizer.param_groups[0]['lr']:.5e}:")
                    self.tb_writer.add_scalar('Loss', loss, (epoch - 1) * len(self.dataloader) + seq_idx)
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
                    self.tb_writer.add_scalar('mini_fitness', mini_metrics['fitness'],
                                         (epoch - 1) * len(self.dataloader) + seq_idx)
                    self.tb_writer.add_scalar('mini_precision', mini_metrics['metrics/precision(B)'],
                                         (epoch - 1) * len(self.dataloader) + seq_idx)
                    self.tb_writer.add_scalar('mini_recall', mini_metrics['metrics/recall(B)'],
                                         (epoch - 1) * len(self.dataloader) + seq_idx)
                    if self.ddp:
                        self.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                        epoch, seq_idx, loss, self.paths['run'], "mini_check.pt")
                    else:
                        self.save_checkpoint(self.model.state_dict(), self.optimizer.state_dict(),
                                             epoch, seq_idx, loss, self.paths['run'], "mini_check.pt")

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
                print(f"Saving checkpoint to {os.path.join(self.paths['model_save'], self.paths['model_name'])}")
                if self.ddp:
                    self.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                    epoch, 0, loss, self.paths['model_save'], self.paths['model_name'])
                else:
                    self.save_checkpoint(self.model.state_dict(), self.optimizer.state_dict(),
                                         epoch, 0, loss, self.paths['model_save'], self.paths['model_name'])

            # Validate
            if self.global_rank == 0:
                metrics = self.validator(model=self.model)
                self.tb_writer.add_scalar('mAP_50', metrics['metrics/mAP50(B)'], epoch)
                self.tb_writer.add_scalar('fitness', metrics['fitness'], epoch)
                self.tb_writer.add_scalar('metrics/precision(B)', metrics['metrics/precision(B)'], epoch)
                self.tb_writer.add_scalar('metrics/recall(B)', metrics['metrics/recall(B)'], epoch)
                # Save Best
                if metrics['fitness'] >= best_metric:
                    print(f"Saving new best to {self.paths['model_save']}")
                    if self.ddp:
                        self.save_checkpoint(self.model.module.state_dict(), self.optimizer.state_dict(),
                                        epoch, 0, loss, self.paths['model_save'], "best.pth")
                    else:
                        self.save_checkpoint(self.model.state_dict(), self.optimizer.state_dict(),
                                             epoch, 0, loss, self.paths['model_save'], "best.pth")

            # Detach tensors
            self.scheduler.step()
        # Cleanup
        self.cleanup()

        # Evalutation
        print("Training Complete:)")

    def save_checkpoint(self, model_dict, opt_dict, epoch, itr, loss, save_path, save_name):
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







