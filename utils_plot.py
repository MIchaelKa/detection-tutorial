import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import numpy as np
import torch

def show_image_and_bb(image, boxes):
    plt.figure(figsize=(6,6))
        
    # img_arr = np.array(image)
    img_arr = np.array(T.ToPILImage()(image))

    image_dims = torch.FloatTensor([img_arr.shape[1], img_arr.shape[0], img_arr.shape[1], img_arr.shape[0]]).unsqueeze(0)

    new_boxes = boxes.clone()
    new_boxes *= image_dims

    color = (255, 0, 0)
    
    for bbox in new_boxes:
        print(bbox)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

    plt.title(f'Shape: {img_arr.shape}', fontsize=16)
    plt.imshow(img_arr)
    plt.grid(False)
    plt.axis('off')

def show_anchors(image, boxes, anchors):
    plt.figure(figsize=(6,6))
    
    img_arr = np.array(T.ToPILImage()(image))

    new_boxes = boxes * 200
    new_anchors = anchors * 200
    
    for bbox in new_boxes:       
        color = (255, 0, 0)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        
    for bbox in new_anchors:       
        color = (0, 0, 255)
        cv2.rectangle(img_arr, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)

    plt.title(f'Shape: {img_arr.shape}', fontsize=16)
    plt.imshow(img_arr)
    plt.grid(False)
    plt.axis('off')