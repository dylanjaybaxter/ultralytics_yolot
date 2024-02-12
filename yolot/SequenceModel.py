'''
File to convert feed forward yolov8 to sequence execution
Author: Dylan Baxter
'''

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import RConv
from ultralytics.utils.loss import v8DetectionLoss
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
        #self.criterion = self.init_criterion()

    def forward(self, x, augnment=False):
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
    def init_criterion(self):
        return SequenceLoss(self)
    
class SequenceLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)

    '''Forward Overwrite with corrected bbox iou loss'''
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    '''bbox iou with corrected ciou loss'''
    def bbox_iou(self, box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """
        Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

        Args:
            box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
            box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
            xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                                (x1, y1, x2, y2) format. Defaults to True.
            GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
            DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
            CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
        """

        # Get the coordinates of bounding boxes
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union
        if CIoU or DIoU or GIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
                if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                    with torch.no_grad():
                        alpha = v / ((1 - iou) + v + eps)
                    return 1/((1-iou) + (rho2 / c2) + (v * alpha))  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
        return iou  # IoU

