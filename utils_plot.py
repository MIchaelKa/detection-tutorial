import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import numpy as np
import torch

from utils import find_jaccard_overlap, generate_anchors, process_anchors

def show_predictions(image, gt_boxes, predicted_boxes, verbose=True):
    img_arr = np.array(T.ToPILImage()(image))

    image_dims = torch.FloatTensor([img_arr.shape[1], img_arr.shape[0], img_arr.shape[1], img_arr.shape[0]]).unsqueeze(0)

    new_boxes = gt_boxes.clone()
    new_boxes *= image_dims
    color = (0, 0, 255)
    
    if verbose:
        print('GT:')
    for bbox in new_boxes:
        if verbose:
            print(bbox)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

    new_boxes = predicted_boxes.clone()
    new_boxes *= image_dims
    color = (255, 0, 0)
    
    if verbose:
        print('Predictions:')
    for bbox in new_boxes:
        if verbose:
            print(bbox)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

    if verbose:
        print(f'Shape: {img_arr.shape}')

    # plt.title(f'Shape: {img_arr.shape}', fontsize=16)
    plt.imshow(img_arr)
    plt.grid(False)
    plt.axis('off')

def show_image_and_bb(image, boxes, verbose=True):
    # img_arr = np.array(image)
    img_arr = np.array(T.ToPILImage()(image))

    image_dims = torch.FloatTensor([img_arr.shape[1], img_arr.shape[0], img_arr.shape[1], img_arr.shape[0]]).unsqueeze(0)

    new_boxes = boxes.clone()
    new_boxes *= image_dims

    color = (255, 0, 0)
    
    for bbox in new_boxes:
        if verbose:
            print(bbox)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

    if verbose:
        print(f'Shape: {img_arr.shape}')

    # plt.title(f'Shape: {img_arr.shape}', fontsize=16)
    plt.imshow(img_arr)
    plt.grid(False)
    plt.axis('off')


def get_top_n_anchors(anchors, gt_boxes, top_n=10):
    # anchors (N1, 4)
    # gt_boxes (N2, 4)
 
    jaccard = find_jaccard_overlap(anchors, gt_boxes) # (N1, N2)
    
    prior_max_iou, prior_gt_box_idx = jaccard.max(1) # (N1), (N1)
    
    _, top_priors_idx = prior_max_iou.sort(descending=True)
    return anchors[top_priors_idx[:top_n]], prior_max_iou[top_priors_idx[:top_n]]

def show_anchors(image, boxes, anchors, verbose=True):
    img_arr = np.array(T.ToPILImage()(image))

    image_dims = torch.FloatTensor([img_arr.shape[1], img_arr.shape[0], img_arr.shape[1], img_arr.shape[0]]).unsqueeze(0)

    new_boxes = boxes.clone()
    new_boxes *= image_dims

    new_anchors = anchors.clone()
    new_anchors *= image_dims
    
    for bbox in new_boxes:       
        color = (255, 0, 0)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        
    for bbox in new_anchors:       
        color = (0, 0, 255)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)

    if verbose:
        print(f'Shape: {img_arr.shape}')

    # plt.title(f'Shape: {img_arr.shape}', fontsize=16)
    plt.imshow(img_arr)
    plt.grid(False)
    plt.axis('off')
    
def show_image_from_dataset(dataset, index, top_n_anchors=10, verbose=True):
    # plt.figure(figsize=(6,6))
    plt.figure(figsize=(10,10))

    image, target = dataset[index]

    if top_n_anchors > 0:

        anchors = generate_anchors()

        print(f'Generating anchors: {anchors.shape}')

        anchor_labels, gt_offsets = process_anchors(anchors, target['boxes'])

        top_n_anchors, top_n_iou = get_top_n_anchors(anchors, target['boxes'], top_n_anchors)

        print(f'Top IoU: {top_n_iou}')

        show_anchors(image, target['boxes'], top_n_anchors, verbose=True)
    else:

        show_image_and_bb(image, target['boxes'], verbose)

        
    label_names = {
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20,   
    }

    if verbose:
        for label in target['labels']:
            print(list(label_names.keys())[label.item()])
            # print(label)

#
# train info
#

def show_loss_history(train_info):
    show_loss(
        train_info['train_loss_history'],
        train_info['train_box_loss_history'],
        train_info['train_cls_loss_history']
    )

def show_loss(loss, box_loss, cls_loss):
    _, axes = plt.subplots(1, 3, figsize=(18,4))

    axes[0].set_title("Loss")
    axes[0].plot(loss)

    axes[1].set_title("Box Loss")
    axes[1].plot(box_loss)

    axes[2].set_title("Cls Loss")
    axes[2].plot(cls_loss)

def show_loss_epochs(train_info):
    _, axes = plt.subplots(1, 3, figsize=(18,4))

    x = np.arange(len(train_info['train_loss_epochs']))

    axes[0].set_title("Loss")
    axes[0].plot(train_info['train_loss_epochs'], '-o')
    axes[0].set_xticks(x)

    axes[1].set_title("Box Loss")
    axes[1].plot(train_info['train_box_loss_epochs'], '-o')
    axes[1].set_xticks(x)

    axes[2].set_title("Cls Loss")
    axes[2].plot(train_info['train_cls_loss_epochs'], '-o')
    axes[2].set_xticks(x)