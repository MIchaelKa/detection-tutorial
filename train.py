import numpy as np

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from faster_rcnn import faster_rcnn
from utils import format_time
from loss import BoxLoss
from transfroms import get_transform

# from dataset import PennFudanDataset
from dataset.pascal_voc_dataset import PascalVOCDataset

import time

def get_device():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    return device

def train_epoch(model, device, criterion, train_loader, optimizer, verbose=True):
    
    t0 = time.time()

    model.train()

    print_every = 100

    for index, (image_batch, target_batch) in enumerate(train_loader):
        
        image_batch = torch.stack(image_batch, dim=0)   
        image_batch = image_batch.to(device)
 
        offsets, labels = model(image_batch)

        loss, box_loss, cls_loss = criterion(labels, offsets, target_batch)

        if verbose and index % print_every == 0:
            print('[train] index: {:>2d}, loss(box/cls) = {:.5f}({:.5f}/{:.5f}) time: {}' \
                .format(index, loss, box_loss, cls_loss, format_time(time.time() - t0)))
        
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

def train_model(model, device, criterion, train_loader, valid_loader, optimizer, num_epochs):
       
    t0 = time.time()
    
    for epoch in range(num_epochs):

        # Train
        train_epoch(model, device, criterion, train_loader, optimizer)

        # Validate
        t2 = time.time()     
        # v_loss_meter, v_score_meter = validate(model, device, valid_loader, criterion)


    train_info = {

    }
 
    return train_info

def run_loader(
    model,
    train_loader,
    valid_loader,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10
    ):

    device = get_device()

    criterion = BoxLoss(device)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )

    train_info = train_model(model, device, criterion, train_loader, valid_loader, optimizer, num_epoch)
    
    return train_info

def create_dataloaders_sampler(
    train_dataset,
    valid_dataset,
    batch_size,
    num_workers=0,
    pin_memory=False
    ):
    
    all_number = len(train_dataset)
    train_number = int(all_number*0.9)

    print(f'data size  all: {all_number}, train: {train_number}')

    train_sampler = SubsetRandomSampler(range(train_number))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    valid_sampler = SubsetRandomSampler(range(train_number, all_number))
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    # TODO: verbose support
    print(f'data_loader_size train: {len(train_loader)}, valid: {len(valid_loader)}')
    
    return train_loader, valid_loader

def main():
    print('run main')

    t1 = time.time()

    # dataset = PennFudanDataset('../PennFudanPed', get_transform(train=False))

    dataset = PascalVOCDataset('./pascal-voc/', 'TRAIN', get_transform(train=True))
    #TODO: valid_dataset = PascalVOCDataset('./pascal-voc/', 'VALID', get_transform(train=False))

    train_loader, valid_loader = create_dataloaders_sampler(dataset, dataset, batch_size=8)

    model = faster_rcnn()

    params = {
        'learning_rate' : 0.001,
        'weight_decay'  : 0,
        'num_epoch'     : 2
    }

    run_loader(model, train_loader, valid_loader, **params)

    print('time: {} '.format(format_time(time.time() - t1)))

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":   
    main()