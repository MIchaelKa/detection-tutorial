import torch
from torchvision.transforms import functional as F

def get_transform(train=True):
    transforms = []
      
    transforms.append(Resize((200, 200)))
    # transforms.append(Scale())
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

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target