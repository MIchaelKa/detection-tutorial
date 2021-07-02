
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import generate_anchors, process_anchors

class BoxLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        # TODO: get from model
        self.anchors = generate_anchors().to(device)
        
    def process_target_batch(self, targets):   
        gt_labels, gt_offsets = [], []
        
        for target in targets:
            # TODO: can we move it all at once?
            gt_boxes = target['boxes'].to(self.device)
            labels, offsets = process_anchors(self.anchors, gt_boxes)
            gt_labels.append(labels)
            gt_offsets.append(offsets)
    #         print(offsets.shape)
            
        gt_labels = torch.stack(gt_labels, dim=0)
        gt_offsets = torch.stack(gt_offsets, dim=0)
                
        return gt_labels, gt_offsets

    def criterion(self, gt_labels, gt_offsets, predicted_labels, predicted_offsets):

        positive_anchors = (gt_labels != 0)

        # v, c = torch.unique(gt_labels, return_counts=True)
        # print(f'anchors pos: {c[1]}, neg: {c[0]}')

        # print(predicted_offsets.shape)
        # print(positive_anchors.shape)
        # print(predicted_offsets[positive_anchors].shape)
        
        # smooth_l1_loss, l1_loss
        box_loss = F.l1_loss(
            predicted_offsets[positive_anchors],
            gt_offsets[positive_anchors],
        )
        
        gt_labels = gt_labels.type_as(predicted_labels)
        
        cls_loss = F.binary_cross_entropy_with_logits(
            predicted_labels,
            gt_labels.unsqueeze(-1)
        )
        
        loss = box_loss + cls_loss
        
        return loss, box_loss, cls_loss

    def forward(self, predicted_labels, predicted_offsets, targets):

        gt_labels, gt_offsets = self.process_target_batch(targets)

        gt_labels = gt_labels.to(self.device)
        gt_offsets = gt_offsets.to(self.device)

        loss = self.criterion(gt_labels, gt_offsets, predicted_labels, predicted_offsets)

        return loss