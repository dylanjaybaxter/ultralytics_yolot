'''
File to convert feed forward yolov8 to sequence execution
Author: Dylan Baxter
'''

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import RConv
import torch


class Args(object):
    pass

class SequenceModel(DetectionModel):
    '''
    This object provides a version of YOLOv8 which gives access to Rconv hidden states and processes entire sequences
    '''
    def __init__(self, cfg="yolov8nT.yaml", device='cpu', verbose=False):
        self.args = Args()
        self.args.cls = 0.5
        self.args.box = 7.5
        self.args.dfl = 1.5
        self.device = 'cpu'
        super().__init__(cfg, verbose=verbose)
        self.device = device
        self.init_criterion()
        self.criterion.to(device)

    def forward(self, x):
        predictions, _ = self.process_sequence(x)
        return predictions

    def process_sequence(self, x, *args, **kwargs):
        '''Note: only batch size of 1 is implemented'''
        # Initialize Output and Hidden State Lists
        hidden_state_list = []
        outputs = []
        inputs = torch.split(x,1,dim=0)
        # Iterate over each input sequence
        for input in inputs:
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
        # Combine outputs into a single "batch"
        outputs_cat = [torch.cat([output[0] for output in outputs], dim=0),
                   torch.cat([output[1] for output in outputs], dim=0),
                   torch.cat([output[2] for output in outputs], dim=0)]

        loss, detached = self.criterion(outputs_cat, sequence_batch)
        return loss / len(outputs)

    def zero_states(self):
        # For each layer
        for layer in self.model:
            # If the layer is one with hidden states
            if type(layer) is RConv:
                # Clear them
                layer.clear_hidden_states()
        return None

    def hidden_states_to(self, device):
        # For each layer
        for layer in self.model:
            # If the layer is one with hidden states
            if type(layer) is RConv:
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
