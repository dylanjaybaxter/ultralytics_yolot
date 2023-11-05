# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.torch_utils import de_parallel
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchvision.ops import nms
from pprint import pprint


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
                                           class_metrics=False, extended_summary=False)


    def validate(self, model):
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
        pbar = tqdm(self.dataloader, desc=pbar_desc, bar_format=bar_format, ascii=False)
        # Iterate through validation data
        average_counter = 0
        for idx, sequence in enumerate(pbar):
            # Forward Pass for sequence
            with autocast(enabled=True):
                outputs = model(sequence[0]['img'].to(self.device))

            # Extract Bounding Boxes for the sequences
            preds = []
            targets = []
            for i in range(sequence[0]['img'].size()[0]):
                # Put Predictions in right format for NMS
                pred = torch.cat([stride.view(1, 144, -1) for stride in outputs[i][0]], dim=2)
                # NMS Call
                filtered_pred = non_max_suppression(pred, conf_thres=self.conf_thres,
                                                    iou_thres=self.iou_thres, classes=[0, 1, 2], max_det=25)

                # Get Truth Boxes
                target_boxes = []
                target_classes = []
                for j in range(sequence[0]['bboxes'].size()[0]):
                    if sequence[0]['frame_idx'][j] == i:
                        box = sequence[0]['bboxes'][j, :]
                        x1 = int((box[0] - 0.5 * box[2]))
                        y1 = int((box[1] - 0.5 * box[3]))
                        x2 = int((box[0] + 0.5 * box[2]))
                        y2 = int((box[1] + 0.5 * box[3]))
                        cls = sequence[0]['cls'][j]
                        target_boxes.append([x1, x2, y1, y2])
                        target_classes.append(cls)
                targets.append({
                    'boxes':sequence[0]['bboxes'][sequence[0]['frame_idx'] == i, :],
                    'labels':sequence[0]['cls'][sequence[0]['frame_idx'] == i].squeeze()
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
                        pred_boxes.append(torch.tensor([x1, y1, x2, y2]).to(self.device))
                        pred_cls.append(torch.tensor(cls).to(self.device))
                        pred_scores.append(torch.tensor(score).to(self.device))
                preds.append({
                    'boxes':torch.clip(torch.stack(pred_boxes, dim=0), min=0, max=1280).unsqueeze(0),
                    'labels':torch.stack(pred_cls, dim=0).to(torch.int).unsqueeze(0),
                    'scores':torch.stack(pred_scores).unsqueeze(0)
                })

            #print(f"Targets({len(targets)}): boxes-{targets[0]['boxes'].shape}, labels-{targets[0]['labels']}")
            #print(f"Preds({len(preds)}): boxes-{preds[0]['boxes']}, labels-{preds[0]['labels']}, scores{preds[0]['scores'].shape}")
            seq_mAP = self.map_op(target=targets, preds=preds)
            #pprint(seq_mAP)

            # Update Progress Bar
            pbar.set_description(f"Seq:{idx+1}/{num_seq} | Acc: {total_acc:.2e}")
            pbar.refresh()

        # Compute Total Metrics and reset internal state of the metric module
        epoch_results = self.map_op.compute()
        self.map_op.reset()

        return epoch_results


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






class DetectionValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.is_coco = False
        self.class_map = None
        self.args.task = 'detect'
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch['img'].shape[2:]
            nb = len(batch['img'])
            bboxes = batch['bboxes'] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch['cls'][batch['batch_idx'] == i], bboxes[batch['batch_idx'] == i]], dim=-1)
                for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        #val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = False
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det)

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=None)  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                None)  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        # Make sure everything is on the same device
        for x in zip(*self.stats):
            x[0].to(self.device)
            x[1].to(self.device)
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot)

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names,
                    on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'],
                    *output_to_target(preds, max_det=self.args.max_det),
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats


def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate trained YOLO model on validation dataset."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = DetectionValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
