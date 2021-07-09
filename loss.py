
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import generate_anchors
from utils import find_jaccard_overlap, cxcy_to_gcxgcy, xy_to_cxcy

class BoxLoss(nn.Module):
    def __init__(self, device, box_loss_settings, anchors):
        super().__init__()

        self.device = device

        self.anchor_threshold = box_loss_settings['anchor_threshold']
        self.fix_no_anchors = box_loss_settings['fix_no_anchors']

        self.enable_hnm = box_loss_settings['enable_hnm']
        self.neg_pos_ratio = box_loss_settings['neg_pos_ratio']

        self.anchors = anchors


    def process_anchors(self, anchors, gt_boxes):
        # anchors (N1, 4)
        # gt_boxes (N2, 4)
    
        jaccard = find_jaccard_overlap(anchors, gt_boxes) # (N1, N2)
        
        max_iou_for_anchors, gt_boxes_id_for_anchors = jaccard.max(1) # (N1), (N1)

        # TODO: test this params using get_positive_anchors
        # get_positive_anchors call here, does it affect performance?
        # using inline? in python?

        if self.fix_no_anchors:
            # fix cases when gt box has no anchor above threshold
            n_objects = gt_boxes.shape[0]
            _, anchor_id_for_object = jaccard.max(0) # (N2), (N2)
            gt_boxes_id_for_anchors[anchor_id_for_object] = torch.LongTensor(range(n_objects)).to(self.device)
            max_iou_for_anchors[anchor_id_for_object] = 1

        positive_anchors_mask = (max_iou_for_anchors > self.anchor_threshold) # (N1)
        gt_boxes_id_for_positive_anchors = gt_boxes_id_for_anchors[positive_anchors_mask] # (n_pos_anchors)

        gt_boxes_for_positive_anchors = gt_boxes[gt_boxes_id_for_positive_anchors]  # (n_pos_anchors, 4)
        positive_anchors = anchors[positive_anchors_mask] # (n_pos_anchors, 4)
        
        gt_offsets = cxcy_to_gcxgcy(xy_to_cxcy(gt_boxes_for_positive_anchors), xy_to_cxcy(positive_anchors)) # (n_pos_anchors, 4)

        prior_labels = positive_anchors_mask.int() # (N1)
        
        return prior_labels, gt_offsets
        
    def process_target_batch(self, targets):   
        gt_labels, gt_offsets = [], []
        
        for target in targets:
            # TODO: can we move it all at once?
            # Should we move it here or later, process_anchors works faster on GPU?
            gt_boxes = target['boxes'].to(self.device)
            labels, offsets = self.process_anchors(self.anchors, gt_boxes)
            gt_labels.append(labels)
            gt_offsets.append(offsets)
            # print(offsets.shape)
            
        gt_labels = torch.stack(gt_labels, dim=0)
        # print(gt_labels.shape)

        gt_offsets = torch.cat(gt_offsets, dim=0)
        # print(gt_offsets.shape)
                
        return gt_labels, gt_offsets

    def criterion(self, gt_labels, gt_offsets, predicted_labels, predicted_offsets):

        positive_anchors = (gt_labels != 0)

        # v, c = torch.unique(gt_labels, return_counts=True)
        # print(f'anchors pos: {c[1]}, neg: {c[0]}')

        # print(predicted_offsets.shape)
        # print(positive_anchors.shape)
        # print(predicted_offsets[positive_anchors].shape)
        
        # smooth_l1_loss, l1_loss
        box_loss = F.smooth_l1_loss(
            predicted_offsets[positive_anchors],
            gt_offsets,
        )
        
        gt_labels = gt_labels.type_as(predicted_labels)

        if self.enable_hnm:
            cls_loss = self.hnm_cls_loss(predicted_labels, gt_labels)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                predicted_labels,
                gt_labels.unsqueeze(-1)
            )

        loss = box_loss + cls_loss
        
        return loss, box_loss, cls_loss

    #
    # Hard Negative Mining
    # from SSD paper
    #
    def hnm_cls_loss(self, predicted_labels, gt_labels):
        positive_anchors = (gt_labels != 0)
        n_anchors = positive_anchors.shape[1]

        cls_loss_all = F.binary_cross_entropy_with_logits(
            predicted_labels,
            gt_labels.unsqueeze(-1),
            reduction='none'
        )
        cls_loss_all = cls_loss_all.squeeze()
        # print(cls_loss.shape)

        n_positives = positive_anchors.sum(dim=1) # per image
        n_hard_negatives = self.neg_pos_ratio * n_positives
        
        cls_loss_pos = cls_loss_all[positive_anchors]
        # print(cls_loss_pos.shape)

        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[positive_anchors] = 0.  # positive priors are ignored (never in top n_hard_negatives)
        cls_loss_neg, _ = cls_loss_neg.sort(dim=1, descending=True)

        hardness_ranks = torch.LongTensor(range(n_anchors)).unsqueeze(0).expand_as(cls_loss_neg).to(self.device)
        # print(hardness_ranks, hardness_ranks.shape)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        cls_loss_hard_neg = cls_loss_neg[hard_negatives]

        # TODO: separate notebook, check shapes
        # print(cls_loss_neg.shape)
        # print(hard_negatives.shape)
        # print(cls_loss_hard_neg.shape)

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        cls_loss = (cls_loss_hard_neg.sum() + cls_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        return cls_loss

    def forward(self, predicted_labels, predicted_offsets, targets):

        gt_labels, gt_offsets = self.process_target_batch(targets)

        gt_labels = gt_labels.to(self.device)
        gt_offsets = gt_offsets.to(self.device)

        loss = self.criterion(gt_labels, gt_offsets, predicted_labels, predicted_offsets)

        return loss