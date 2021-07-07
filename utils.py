import torch

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h
                      
def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def calculate_mAP(det_boxes, det_scores, true_boxes, true_labels, device, threshold=0.5):

    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    # n_objects is the total no. of objects across all images
    true_images = torch.LongTensor(true_images).to(device) # (n_objects)
    true_boxes = torch.cat(true_boxes, dim=0).to(device)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0).to(device)  # (n_objects)

    det_images = list()
    for i in range(len(det_scores)):
        det_images.extend([i] * det_scores[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    n_objects = true_boxes.size(0)
    true_boxes_detected = torch.zeros(n_objects, dtype=torch.uint8).to(device)

    # Sort detections in decreasing order of confidence/scores
    det_scores_sorted, sort_ind = torch.sort(det_scores, dim=0, descending=True)  # (n_detections)
    det_images_sorted = det_images[sort_ind]  # (n_detections)
    det_boxes_sorted = det_boxes[sort_ind]  # (n_detections, 4)

    n_detections = det_boxes.size(0)

    # In the order of decreasing scores, check if true or false positive
    true_positives = torch.zeros((n_detections), dtype=torch.float).to(device)  # (n_detections)
    false_positives = torch.zeros((n_detections), dtype=torch.float).to(device)  # (n_detections)
    false_negatives = torch.zeros((n_detections), dtype=torch.float).to(device)  # (n_detections)

    for d in range(n_detections):
        this_detection_box = det_boxes_sorted[d].unsqueeze(0)  # (1, 4)
        this_image = det_images_sorted[d]  # (), scalar
        
        # Find objects in the same image and whether they have been detected before  
        this_image_boxes = true_boxes[true_images==this_image] # (n_objects_in_img, 4)
        
        # Find maximum overlap of this detection with objects in this image of this class
        overlaps = find_jaccard_overlap(this_detection_box, this_image_boxes)  # (1, n_objects_in_img)
        max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
        
        # Index in the true_boxes and true_boxes_detected to find duplicated detections
        original_ind = torch.LongTensor(range(true_boxes.size(0)))[true_images == this_image][ind]
        
        if max_overlap > threshold:      
            if true_boxes_detected[original_ind] == 0:
                true_positives[d] = 1
                true_boxes_detected[original_ind] = 1
            else:
                false_positives[d] = 1
        else:
            false_positives[d] = 1
            
        false_negatives[d] = (1-true_boxes_detected).sum()

    # Compute cumulative precision and recall at each detection in the order of decreasing scores
    cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_detections)
    cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_detections)

    cumul_precision = cumul_true_positives / (
        cumul_true_positives + cumul_false_positives + 1e-10)  # (n_detections)

    cumul_recall = cumul_true_positives / (
        cumul_true_positives + false_negatives + 1e-10)  # (n_detections)

    # AP calculations
    recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)

    precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
    for i, t in enumerate(recall_thresholds):
        recalls_above_t = cumul_recall >= t
        if recalls_above_t.any():
            precisions[i] = cumul_precision[recalls_above_t].max()
        else:
            precisions[i] = 0.

    average_precision = precisions.mean()

    return average_precision

#
# anchors
#

import math

def get_top_n_anchors(anchors, gt_boxes, top_n=10):
    # anchors (N1, 4)
    # gt_boxes (N2, 4)
 
    jaccard = find_jaccard_overlap(anchors, gt_boxes) # (N1, N2)
    
    prior_max_iou, prior_gt_box_idx = jaccard.max(1) # (N1), (N1)
    
    _, top_priors_idx = prior_max_iou.sort(descending=True)
    return anchors[top_priors_idx[:top_n]], prior_max_iou[top_priors_idx[:top_n]]

def generate_anchors(clip=False):

    # TODO: anchors per pixel, return?
    
    # feature_map_size = 7
    feature_dims = [25, 13, 7, 4]

    # Difference between SSD and RPN
    #
    # SSD has more accurate setup of anchor generation
    # + support of different numbers of anchors per pixel of feature map

    # scales = [0.9, 0.6, 0.3]

    feature_map_scales = [0.2, 0.4, 0.6, 0.8]

    aspect_ratios = [1., 2., 0.5]
    
    anchors = []
    
    for idx, f in enumerate(feature_dims):
        scale = feature_map_scales[idx]
        for i in range(f):
            for j in range(f):
                # for scale in scales:
                for ratio in aspect_ratios:
                    x = (i + 0.5) / f
                    y = (j + 0.5) / f
                    width = scale * math.sqrt(ratio)
                    height = scale / math.sqrt(ratio)
                    
                    anchor = [x, y, width, height]
                    anchors.append(anchor)

    # TODO: torch.tensor here?
    # - do we already need torch.tensor here?
    # - only if we use it in loss calculation right from her
    # return np.array(anchors)

    # TODO: clamp here?
    # before or after cxcy_to_xy?
    # no sence to make it before?

    anchors = torch.tensor(anchors)
    anchors = cxcy_to_xy(anchors)
    if clip:
        anchors.clamp_(0, 1)
    return anchors
    

#
# time
#

import datetime

## TODO time func

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#
# common
#

import numpy as np
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
