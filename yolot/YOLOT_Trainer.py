'''
This file contains a class used for training yolot
'''
# General Imports
import datetime

import torch
import os

# Local Imports
from yolot.BMOTSDataset import BMOTSDataset, collate_fn
from yolot.YOLOT_Validator import SequenceValidator
from yolot.SequenceModel import SequenceModel
import torch.optim as opt
from tqdm import tqdm
from ultralytics.utils.ops import non_max_suppression
from torch.cuda.amp import autocast, GradScaler

# Parallelization
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics.data.build import InfiniteDataLoader
from torch.optim.lr_scheduler import LambdaLR

# Logging
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter


class YolotTrainer():
    def __init__(self, cfg):
        '''
        :param cfg: a dict that contains the following feilds
        - model (file path of the model .yaml)
        - epochs (default epochs)
        - data (path to dataset)
        - workers (number of workers for dataloader)
        - met_save_path (path to save metrics to)
        - visualize (Bool that determines whether images will be displayed)
        - seq_len (length of sequences to train on)
        - cls (class loss multiplier)
        - box (box loss multiplier)
        - dfl (other loss multipler)
        - lr0 (intial learning rate)
        - DEBUG (debug flag)
        - log_port (tensorboard port)
        - run_name (name of folder to save/load progress and results in)
        - seq_cap (max number of sequences per epoch, for debug)
        - save_freq (number of sequences between saving checkpoints)
        '''
        self.cfg = cfg
        if cfg['devices'] > 1:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            if cfg['verbose']:
                print(f"Training on GR: {self.global_rank}/{self.world_size}, LR: {self.local_rank}...checking in...")
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1

        # Setup Device
        if torch.cuda.is_available():
            self.device = 'cuda:' + str(self.local_rank)
        else:
            self.device = torch.device('cpu')
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        if cfg['verbose']:
            self.print_cuda_info()

        # Setup Paths
        self.run_name = cfg['run_name']
        self.model_save_path = os.path.join(cfg['met_save_path'], cfg['run_name'], "weights", "checkpoint.pth")
        self.tb_save_path = os.path.join(cfg['met_save_path'], cfg['run_name'], "tb", )
        self.model_load_path = self.create_file_structure(self.model_save_path, self.run_name)

        # Setup Dataloader
        self.train_loader, self.val_loader = self.init_dataloader(cfg['data'], cfg['seq_len'], cfg['workers'], cfg['shuffle'])
        # Create Model
        self.model = self.init_model(cfg['model'])
        class Args(object):
            pass
        self.model.args = Args()
        self.model.args.cls = cfg['cls']
        self.model.args.box = cfg['box']
        self.model.args.dfl = cfg['dfl']
        # Initialize grad scaler, optimizer, and scheduler
        self.scaler = GradScaler(enabled=True)
        self.optimizer = opt.SGD(self.model.parameters(), lr=cfg['lr0'], momentum=0.9)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lambda epoch: (0.9 ** epoch)])
        # Initialize Validator
        self.validator = SequenceValidator(dataloader=self.val_loader, device='cpu')
        # Load Checkpoint if one exists



    def init_model(self, cfg_path, model_load_path):
        model = SequenceModel(cfg=cfg_path, device=self.device, verbose=(self.local_rank == 0))
        if os.path.exists(self.model_load_path):
            ckpt = torch.load(self.model_load_path)
            model.load_state_dict(ckpt['model'], strict=False)
        if self.ddp:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)



    def init_dataloader(self, dataset_path, seq_len, workers, shuffle=False):
        # Create Datasets for Training and Validation
        training_dataset = BMOTSDataset(dataset_path, "train", device=self.device, seq_len=seq_len)
        val_dataset = BMOTSDataset(dataset_path, "val", device=self.device, seq_len=seq_len)
        # Create Samplers for distributed processing
        train_sampler = DistributedSampler(training_dataset, shuffle=shuffle,
                                           drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=shuffle,
                                         drop_last=False)
        # Use Datasets to Create Autoloader
        train_loader = InfiniteDataLoader(training_dataset, num_workers=workers, batch_size=1, shuffle=False,
                                          collate_fn=collate_fn, drop_last=False, pin_memory=False,
                                          sampler=train_sampler)
        val_loader = InfiniteDataLoader(val_dataset, num_workers=workers, batch_size=1, shuffle=False,
                                        collate_fn=collate_fn, drop_last=False, pin_memory=False, sampler=val_sampler)
        return train_loader, val_loader

    def init_tensorboard(self, dir, port):
        '''
        Initializes tensorboard
        :param dir:
        :param port:
        :return:
        '''
        # Initialize Tensorboard
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(dir, dt)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port), '--bind_all'])
        url = tb.launch()
        print(f"Tensorboard started listening to {log_dir} and broadcasting on {url}")
        tb_writer = SummaryWriter(log_dir=log_dir)
        return tb_writer

    def create_file_structure(self, save_path, run_name):
        if os.path.exists(os.path.join(save_path, run_name)):
            # Look for checkpoint
            print(f"Continuing Run: {run_name}")
            if os.path.exists(os.path.join(save_path, run_name, "weights", "checkpoint.pth")):
                model_load_path = os.path.join(save_path, run_name, "weights", "checkpoint.pth")
                if self.verbose:
                    print("Using previous checkpoint...")
            else:
                model_load_path = ""
                if self.verbose:
                    print("Starting model from scratch")
        else:
            # Create new file structure
            print(f"Creating new run: {run_name}")
            os.mkdir(os.path.join(save_path, run_name))
            os.mkdir(os.path.join(save_path, run_name, "weights"))
            model_load_path = ""
        return model_load_path


    def print_cuda_info(self):
        print(torch.__version__)
        print(torch.cuda.nccl.is_available(torch.randn(1).cuda()))
        print(torch.cuda.nccl.version())