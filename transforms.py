import random
import torch
from torchvision.transforms import functional as F

def get_transform(train=True):
    transforms = []
      
    transforms.append(Resize((200, 200)))
    # transforms.append(Scale())

    transforms.append(ToTensor())
    transforms.append(Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]))
    
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)

def get_transform_to_show():
    transforms = []
      
    transforms.append(Scale())
    transforms.append(ToTensor())

    return Compose(transforms)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image, target):
        old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        size_tensor = torch.FloatTensor([self.size[0], self.size[1], self.size[0], self.size[1]]).unsqueeze(0)
        scale_factors = old_dims / size_tensor        
        
        image = F.resize(image, self.size)
        
        boxes = target["boxes"]
        
        new_boxes = boxes / scale_factors
        new_boxes = new_boxes / size_tensor
        
        target["boxes"] = new_boxes
    
        return image, target

class Scale(object):
    # TODO: do this once, before training       
    def __call__(self, image, target):
        image_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        boxes = target["boxes"]    
        new_boxes = boxes / image_dims 
        target["boxes"] = new_boxes
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = 1 - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target