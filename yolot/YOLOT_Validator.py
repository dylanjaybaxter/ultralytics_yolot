# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import torch
from tqdm import tqdm
from ultralytics.utils.ops import non_max_suppression
from torchmetrics.detection import MeanAveragePrecision

# Added
from torch.cuda.amp import autocast


class SequenceValidator():
    def __init__(self, dataloader, iou_thres=0.8, conf_thres=0.8, class_dict=None, device='cpu', tb_writer=None):
        self.dataloader = dataloader
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.class_dict = class_dict
        self.device = device
        self.iou_op = IntersectionOverUnion(box_format='xyxy', iou_threshold=iou_thres,
                                            class_metrics=True, respect_labels=True).to(device)
        self.map_op = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.25,0.5,0.75,0.95],
                                           class_metrics=False, extended_summary=False).to(device)
        self.global_rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.running_metrics = {}

    def validate(self, model):
        with torch.no_grad():
            # Setup Metrics
            self.metrics = {}
            total_acc = 0
            total_fp = 0

            # Set model to val
            model.eval()

            # Setup Progress Bar
            # Set Up Loading bar for epoch
            bar_format = f"::Val|{{bar:30}}| {{percentage:.2f}}% | [{{elapsed}}<{{remaining}}] | {{desc}}"
            num_seq = len(self.dataloader)
            pbar_desc = f'Seq:0/{num_seq} | Acc: {total_acc:.2e}'
            pbar = tqdm(self.dataloader, desc=pbar_desc, bar_format=bar_format, ascii=False, disable=(self.global_rank != 0))
            # Iterate through validation data
            metric_counter = 50
            for idx, sequence in enumerate(pbar):
                # Clear hidden states
                model.module.zero_states()

                # Forward Pass for sequence
                with autocast(enabled=True):
                    outputs = model(sequence[0]['img'])

                # Extract Bounding Boxes for the sequences
                preds = []
                targets = []
                for i in range(sequence[0]['img'].size()[0]):
                    # Put Predictions in right format for NMS
                    pred = torch.cat([stride.view(1, 144, -1) for stride in outputs[i][0]], dim=2)
                    # NMS Call
                    filtered_pred = non_max_suppression(pred.detach(), conf_thres=self.conf_thres,
                                                        iou_thres=self.iou_thres, classes=[0, 1, 2], max_det=25)

                    targets.append({
                        'boxes':sequence[0]['bboxes'].reshape(-1,4)[sequence[0]['frame_idx'] == i, :].detach().clone().reshape(-1,4).to(self.device),
                        'labels':sequence[0]['cls'][sequence[0]['frame_idx'] == i].detach().clone().reshape(-1).to(self.device)
                    })

                    # Get predicted boxes
                    pred_boxes = []
                    pred_cls = []
                    pred_scores = []
                    for pred_ind in range(filtered_pred[0].size()[0]):
                        # Check for class 1
                        box = filtered_pred[0][pred_ind, 0:4]
                        cls = filtered_pred[0][pred_ind, 5]
                        score = filtered_pred[0][pred_ind, 4]
                        x1 = int((box[0] - 0.5 * box[2]))
                        y1 = int((box[1] - 0.5 * box[3]))
                        x2 = int((box[0] + 0.5 * box[2]))
                        y2 = int((box[1] + 0.5 * box[3]))
                        # Only assign if box is in correct format
                        if((x1 < x2) and (y1 < y2)):
                            pred_boxes.append(torch.tensor([x1, y1, x2, y2]))
                            pred_cls.append(cls.to(torch.int))
                            pred_scores.append(score)
                    preds.append({
                        'boxes': torch.clip(torch.stack(pred_boxes, dim=0), min=0, max=1280).detach().clone().to(self.device),
                        'labels': torch.stack(pred_cls, dim=0).detach().clone().to(self.device),
                        'scores': torch.stack(pred_scores).detach().clone().to(self.device)
                    })
                # Calculate mAP of sequence
                seq_mAP = self.map_op(target=targets, preds=preds)

                # Calculate Running Averages
                if idx == 0:
                    self.init_metrics(seq_mAP)
                else:
                    self.update_metrics(seq_mAP, idx)

                # Reset Targets and Predictions and hopefully free tensors
                targets = None
                preds = None

                # Update Progress Bar
                if self.global_rank == 0 or (idx+1) == num_seq:
                    pbar.set_description(f"Seq:{idx+1}/{num_seq} | Acc: {seq_mAP['map_50']:.2e}, Running: {self.metrics['map_50']:.2e}")
                    pbar.refresh()

            self.map_op.reset()

            return self.metrics


    def determine_overlap(self, tbox, pbox):
        '''
        This function determines the overlap on a single dimension between two sets
        of points in tbox and pbox
        :param tbox:
        :param pbox:
        :return: overlapping area
        '''
        # x overlap
        t_min_x = min(tbox[0], tbox[1])
        t_max_x = max(tbox[0], tbox[1])
        p_min_x = min(pbox[0], pbox[1])
        p_max_x = max(pbox[0], pbox[1])
        if t_max_x > p_min_x:
            overlap = t_max_x - p_min_x
        elif p_max_x > t_min_x:
            overlap = p_max_x - t_min_x
        else:
            overlap = 0
        return overlap

    def init_metrics(self, meas):
        '''
        Creates fields in the metrics dict based on a target dict and initializes them
        :param meas:
        :return:
        '''
        for key in meas.keys():
            if meas[key].size() == torch.Size([]):
                self.metrics[key] = meas[key]

    def update_metrics(self, new_meas, idx):
        '''
        Updates metrics as running average in the metrics dict if they exist and have same key as target
        :param new_meas:
        :param idx:
        :return:
        '''
        for key in new_meas.keys():
            if new_meas[key].size() == torch.Size([]):
                if key in self.metrics and new_meas[key] and new_meas[key] >= 0:
                    self.metrics[key] = ((self.metrics[key] * idx) + new_meas[key])/(idx+1)

    def determine_overlap_2d(self, tbox, pbox):
        '''
        Determine the overlap and iou of a pair of 2D bounding boxes in xyxy format
        :param tbox:
        :param pbox:
        :return:
        '''
        # Determine overlapping regions
        x_overlap = self.determine_overlap([tbox[0], tbox[2]], [pbox[0], pbox[2]])
        y_overlap = self.determine_overlap([tbox[1], tbox[3]], [pbox[1], pbox[3]])
        A_overlap = x_overlap*y_overlap
        At = (tbox[2]-tbox[0])*(tbox[3]-tbox[1])
        Ap = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
        A_union = At+Ap-A_overlap
        return A_overlap, A_overlap/A_union