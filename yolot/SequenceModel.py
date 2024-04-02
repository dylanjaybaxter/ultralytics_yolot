'''
File to convert feed forward yolov8 to sequence execution
Author: Dylan Baxter
'''

import torch
from torch import nn
from copy import deepcopy
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics.utils.plotting import feature_visualization
from yolot.rnn import RConv, ConvGRU, Rnn, AddRnn
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment)
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import (initialize_weights, make_divisible)
from ultralytics.nn.tasks import yaml_model_load
import contextlib
from yolot.loss import SequenceLoss


class Args(object):
    pass

class SequenceModel(BaseModel):
    '''
    This object provides a version of YOLOv8 which gives access to Rconv hidden states and processes entire sequences
    '''
    def __init__(self, cfg="yolov8nT.yaml", ch=3, nc=None, verbose=True, device='cpu'):  # model, input channels, number of classes
        super().__init__()
        # Set Device to CPU for initialization
        self.device = 'cpu'

        if cfg[-6] in 'nmlx':
            scale = cfg[-6]
            cfg = cfg[:-6]+cfg[-5:]
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        self.yaml['scale'] = scale

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model_custom(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))[0]])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

        self.device = device

    def forward(self, x, augment=False, visualize=False, embed=False):
        predictions, _ = self.process_sequence(x)
        return predictions
    
    def _predict_once(self, x, profile=False, visualize=False, embed=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []
        hidden_states = [] # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if type(m) is RConv or type(m) is ConvGRU:
                x, hidden_state = m(x)
                hidden_states.append(hidden_state)
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        if len(hidden_states) == 0:
            return x
        else:
            return x, hidden_states

    def process_sequence(self, x, *args, **kwargs):
        '''
        Input:
        - torch.tensor (seq_len, batch, ch, w, h)
        Output:
        - List of outputs and hidden states
        '''
        # Initialize Output and Hidden State Lists
        hidden_state_list = []
        outputs = []
        inputs = torch.split(x,1,dim=0)
        # Iterate over each input sequence
        for input in inputs:
            # Cut out sequence dimension if it exists
            if len(input.size()) > 4:
                input = input.squeeze(0)
            # Feed forward slice and save hidden state
            y, self.hidden_states = super().forward(input.to(self.device))
            # Save output and hidden state
            hidden_state_list.append(self.hidden_states)
            outputs.append(y)

        # Output hidden states and model output
        return outputs, hidden_state_list

    def sequence_loss(self, outputs, sequence_batch):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        # Find Batch Size
        bs = outputs[0][0].size()[0]
        # Combine outputs into a single "batch"
        outputs_cat_list = []
        for batch_id in range(bs):
            outputs_cat_list.append([torch.cat([output[0][batch_id,:,:,:].unsqueeze(0) for output in outputs], dim=0),
                                torch.cat([output[1][batch_id,:,:,:].unsqueeze(0) for output in outputs], dim=0),
                                torch.cat([output[2][batch_id,:,:,:].unsqueeze(0) for output in outputs], dim=0)])
        outputs_cat = [torch.cat([output[0] for output in outputs_cat_list], dim=0),
                        torch.cat([output[1] for output in outputs_cat_list], dim=0),
                        torch.cat([output[2] for output in outputs_cat_list], dim=0)]

        loss, detached = self.criterion(outputs_cat, sequence_batch)
        return loss / len(outputs)

    def zero_states(self):
        # For each layer
        for layer in self.model:
            # If the layer is one with hidden states
            if type(layer) is RConv:
                layer.clear_hidden_states()
            elif type(layer) is ConvGRU:
                layer.clear_hidden_states()
        return None

    def hidden_states_to(self, device):
        # For each layer
        for layer in self.model:
            # If the layer is one with hidden states
            if type(layer) is RConv:
                layer.hidden_states_to(device)
            if type(layer) is ConvGRU:
                layer.hidden_states_to(device)
        return None

    def model_to(self, device):
        self.device = device
        self.to(device)
        self.hidden_states_to(device)

    def get_hidden_states(self):
        # For each layer
        hidden_states = []
        for layer in self.model:
            # If the layer is one with hidden states
            if type(layer) is RConv:
                hidden_states.append(layer.get_hidden_state())
        return hidden_states
    
    def init_criterion(self):
        return SequenceLoss(self)

def parse_model_custom(d, ch, verbose=True, batch=1):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    ratio = 2 # Placeholder until ratio can be identified
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is Rnn:
            args[0][1] = args[0][1] * width
            args[1][1] = args[1][1] * width
            c2 = ch[f]
        elif m is AddRnn:
            c2 = ch[f]
        elif m is RConv:
            args[0] = ch[f]
            c2 = ch[f]
        elif m is ConvGRU:
            args[0] = ch[f]
            c2 = ch[f]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)