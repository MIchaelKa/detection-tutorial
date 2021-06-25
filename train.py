import numpy as np

import torch

from faster_rcnn import faster_rcnn
from utils import format_time
from dataset import PennFudanDataset
from loss import BoxLoss

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
    
    model.train()
    
    t0 = time.time()

    for index, (image_batch, target_batch) in enumerate(train_loader):
        
        image_batch = torch.stack(image_batch, dim=0)   
        image_batch = image_batch.to(device)
 
        offsets, labels = model(image_batch)

        loss = criterion(labels, offsets, target_batch)

        if verbose:
            print('[train] index: {:>2d}, loss = {:.5f}, time: {}' \
                .format(index, loss, format_time(time.time() - t0)))
        
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

def main():
    print('main')

    t1 = time.time()

    dataset = PennFudanDataset('../PennFudanPed', get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # img, target = dataset[0]
    images, targets = next(iter(data_loader))
    # images = next(iter(data_loader))

    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]

    net = faster_rcnn()

    x = torch.rand(5, 3, 300, 300)

    images = torch.stack(images, dim=0)

    out = net(images)

    # print(out.shape)

    print('time: {} '.format(format_time(time.time() - t1)))

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":   
    main()