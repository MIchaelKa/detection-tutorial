import torch
import torch.nn as nn
# from torch import nn
from torch.nn import functional as F

from torchvision import models

from utils import generate_anchors, find_jaccard_overlap, gcxgcy_to_cxcy, xy_to_cxcy, cxcy_to_xy

def faster_rcnn(device):
    # remember about 7x7 first conv, does resnext have it?
    resnet = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(resnet.children())[:-2])

    anchors = generate_anchors().to(device)

    return FasterRCNN(backbone, anchors, device).to(device)


class RoIPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class RPN(nn.Module):

    def __init__(self):
        super().__init__()

        # TODO: all consts to init

        in_channels = 512
        intermediate_size = in_channels

        self.conv3 = nn.Conv2d(in_channels, intermediate_size, kernel_size=3, stride=1, padding=1)

        N = 9 # TODO: anchors per pixel
        self.reg = nn.Conv2d(intermediate_size, N*4, kernel_size=1, stride=1, padding=0)
        self.cls = nn.Conv2d(intermediate_size, N, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x,):

        x = F.relu(self.conv3(x))

        boxes = self.reg(x)
        classes = self.cls(x)

        return boxes, classes


class FasterRCNN(nn.Module):
    def __init__(self, backbone, anchors, device):
        super().__init__()
        self.backbone = backbone
        self.anchors = anchors
        self.device = device
        self.rpn = RPN()

    def forward(self, x):
        x = self.backbone(x)

        box, cls = self.rpn(x)

        # view will not work without contiguous, but reshape should work
        # should we use reshape here after permute?!
        # box = box.reshape(box.shape[0], -1, 4)
        # box = box.reshape(box.shape[0], 4, -1)

        box = box.permute(0, 2, 3, 1).contiguous()
        box = box.view(box.shape[0], -1, 4)

        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(cls.shape[0], -1, 1)
         
        return box, cls

    def detect(self, offsets, labels, prob_threshold=0.5, max_overlap=0.5):

        batch_size = offsets.shape[0]

        detections = []
        confidences = []

        for i in range(batch_size):
            image_labels = labels[i].squeeze()
            image_offsets = offsets[i]

            labels_probs = torch.sigmoid(image_labels)

            positive_indices = labels_probs > prob_threshold

            num_positives = positive_indices.sum()
            # print(f'[FasterRCNN] num_positives: {num_positives}')

            positive_offsets = image_offsets[positive_indices]
            positive_anchors = xy_to_cxcy(self.anchors[positive_indices])
            positive_scores = labels_probs[positive_indices]

            predicted_boxes = cxcy_to_xy(gcxgcy_to_cxcy(positive_offsets, positive_anchors))

            sorted_scores, indices = torch.sort(positive_scores, descending=True)
            sorted_boxes = predicted_boxes[indices]

            overlap = find_jaccard_overlap(sorted_boxes, sorted_boxes)

            # Non-Maximum Suppression (NMS)

            # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
            # 1 implies suppress, 0 implies don't suppress
            suppress = torch.zeros((num_positives), dtype=torch.uint8).to(self.device)  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(num_positives):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                # Find such boxes and update suppress indices
                suppress = torch.max(suppress, overlap[box] > max_overlap)
                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            final_boxes = sorted_boxes[1-suppress]
            final_scores = sorted_scores[1-suppress]

            detections.append(final_boxes)
            confidences.append(final_scores)
    
        return detections, confidences

# net = faster_rcnn()
# print(net)