# Author: Dylan Baxter, 8/17/2023, Cal Poly San Luis Obispo
'''

Recurrent Modules

'''
# Imports
import math

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from .conv import Conv
import torchvision.transforms.functional as tfunc
import torch.nn.functional as func

# Module Definitions
class Rnn(nn.Module):
    default_act = nn.Sigmoid

    def __init__(self, input_shape, output_shape, hidden_size=8, act=True):
        super().__init__()
        self.size = np.cumprod(input_shape)[-1].astype(int)
        self.rnn = nn.RNN(input_size=self.size, hidden_size=hidden_size)
        self.act = self.default_act if act is True else nn.Identity
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.resize_output = True if output_shape != input_shape else False
        self.hidden_state = torch.ones(1,hidden_size)

    def forward(self, x):
        # Flatten the matrix
        x = x.flatten().reshape(1,-1)
        # Run the resulting vector through the RNN
        y, self.hidden_state = self.rnn(x, self.hidden_state)
        # Reshape Output to original shape
        y = np.reshape(y, self.input_shape)
        # Resize for output if needed
        if self.resize_output:
            y = np.resize(y,self.output_shape)
        return y

class AddRnn(nn.Module):
    default_act = nn.Sigmoid

    def __init__(self, input_shape, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.rnn_output_shape = deepcopy(input_shape)
        self.rnn_output_shape[-2:] = [i*2/scaling_factor for i in self.rnn_output_shape[-2:]]
        self.hidden_size = np.cumprod(self.rnn_output_shape)[-1].astype(int)
        self.rnn = Rnn(input_shape=input_shape, output_shape=self.rnn_output_shape, hidden_size=self.hidden_size)
        self.upscale = nn.Upsample(scaling_factor, mode='nearest')

    def forward(self, x):
        '''Forward Function for AddRnn Module'''
        return x + self.upscale(self.rnn(x))


class RConv(nn.Module):
    default_act = nn.SiLU

    def __init__(self, ch, hidden_size, k, batch_size=1, device='cpu'):
        super().__init__()
        self.hidden_size = [batch_size, ch, hidden_size, hidden_size]
        self.conv = Conv(ch, ch, k=k, s=2)
        self.hidden_states = []
        self.device = device
        for i in range(3):
            self.hidden_states.append(
                torch.ones(self.hidden_size, requires_grad=True).to(self.device))

    def forward(self, x):
        # Pass input through convolutional layer
        x_conv = self.conv(x)
        x_compressed = tfunc.resize(x_conv, self.hidden_size[3:])
        # Create Output Tensor
        expanded = self.expand_tensors(self.hidden_states, x_conv.shape)
        y = torch.cat(expanded+[x_conv], dim=3)
        y_reshaped = torch.reshape(y, x.shape)
        # Save compressed input to hidden state by adding it to FIFO Queue
        self.hidden_states = [x_compressed] + self.hidden_states[1:]
        # Return Output
        return y_reshaped, self.hidden_states

    def get_hidden_state(self):
        return self.hidden_states

    def hidden_states_to(self, device):
        self.device = device
        for i in range(3):
            self.hidden_states[i] = self.hidden_states[i].to(device)

    def clear_hidden_states(self):
        for i in range(3):
            self.hidden_states[i] = \
                torch.ones(self.hidden_size, requires_grad=True).to(self.device)
            self.hidden_states[i] = self.hidden_states[i].detach().to(self.device)
        return None

    def expand_tensors(self, tensor_list, target_size):
        expanded = []
        for t in tensor_list:
            expanded.append(tfunc.resize(t, target_size[3:]))
        return expanded



'''
Convolutional Gated Recurrent Unit based on https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
And adapted for use in yolot
'''

class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, k, b=1, device='cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Members
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.b = b

        # Layers
        self.reset = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size=k, padding=k//2)
        self.update = nn.Conv2d(input_size+hidden_size, hidden_size, k, padding=k//2)
        self.out = nn.Conv2d(input_size+hidden_size, hidden_size, k, padding=k//2)
        self.bottleneck = nn.Conv2d(hidden_size, input_size, 1)

        # Hidden State
        self.device = device
        self.hidden_state = None

        # Initialization
        nn.init.orthogonal(self.reset.weight)
        nn.init.orthogonal(self.update.weight)
        nn.init.orthogonal(self.out.weight)
        nn.init.orthogonal(self.bottleneck.weight)
        nn.init.constant(self.reset.bias, 0.)
        nn.init.constant(self.reset.bias, 0.)
        nn.init.constant(self.reset.bias, 0.)
        nn.init.constant(self.bottleneck.bias, 0.)
    
    def forward(self, x):
        '''
        Conv GRU Function
        Input Tensor x: (batch, ch, h, w)
        Output:
        '''
        # Initialize Hidden State if Needed
        if self.hidden_state is None:
            self.spatial = x.data.size()[2:]
            self.hidden_state = torch.zeros(
                [self.b, self.hidden_size]+list(self.spatial), 
                requires_grad=True).to(self.device)
        
        # Conv GRU Function
        full_state = torch.cat([x, self.hidden_state], dim=1)
        update_mask = func.sigmoid(self.update(full_state))
        reset_mask = func.sigmoid(self.reset(full_state))
        canidates = func.tanh(self.out(torch.cat([x,self.hidden_state*reset_mask], dim=1)))
        self.hidden_state = canidates * update_mask + self.hidden_state * (1 - update_mask)

        # Simple Reduction of output channels to match input channels
        y = self.bottleneck(self.hidden_state)
        return y, self.hidden_state

    def get_hidden_states(self):
        return self.hidden_state

    def hidden_states_to(self, device):
        self.device = device
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.to(device)

    def clear_hidden_states(self):
        self.hidden_state = None

