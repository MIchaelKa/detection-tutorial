import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from faster_rcnn import faster_rcnn
from utils import format_time
from loss import BoxLoss
from transforms import get_transform
from metrics import AverageMeter

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

def collate_fn(batch):
    return tuple(zip(*batch))

def create_dataloaders_sampler(
    train_dataset,
    valid_dataset,
    batch_size,
    num_workers=0,
    pin_memory=False,
    verbose=True,
    debug=False
    ):
    
    all_number = len(train_dataset)
    if debug:
        all_number = 12

    train_number = int(all_number*0.9)

    if verbose:
        print(f'data size, all: {all_number}, train: {train_number}')

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

    if verbose:
        print(f'data loader size, train: {len(train_loader)}, valid: {len(valid_loader)}, batch_size {batch_size}')
    
    return train_loader, valid_loader

def train_epoch(model, device, criterion, train_loader, optimizer, verbose=True):
    
    t0 = time.time()
    print_every = 20

    model.train()

    loss_meter = AverageMeter()
    box_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()

    for index, (image_batch, target_batch) in enumerate(train_loader):
        
        image_batch = torch.stack(image_batch, dim=0)   
        image_batch = image_batch.to(device)
 
        offsets, labels = model(image_batch)

        loss, box_loss, cls_loss = criterion(labels, offsets, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update meters
        loss_meter.update(loss.item())
        box_loss_meter.update(box_loss.item())
        cls_loss_meter.update(cls_loss.item())

        if verbose and index % print_every == 0:
            print('[train] index: {:>2d}, loss(box/cls) = {:.5f}({:.5f}/{:.5f}) time: {}' \
                .format(
                    index,
                    loss_meter.compute_average(),
                    box_loss_meter.compute_average(),
                    cls_loss_meter.compute_average(),
                    format_time(time.time() - t0))
                )

    return (loss_meter, box_loss_meter, cls_loss_meter)

def train_model(model, device, criterion, train_loader, valid_loader, optimizer, num_epochs, verbose=True):
       
    t0 = time.time()

    train_loss_history = []
    train_box_loss_history = []
    train_cls_loss_history = []
    
    train_loss_epochs = []
    train_box_loss_epochs = []
    train_cls_loss_epochs = []

    if verbose:
        print('training started...')
    
    for epoch in range(num_epochs):

        # Train
        t1 = time.time()
        loss_meters = train_epoch(model, device, criterion, train_loader, optimizer)

        loss_meter, box_loss_meter, cls_loss_meter = loss_meters

        train_loss_history.extend(loss_meter.history)
        train_box_loss_history.extend(box_loss_meter.history)
        train_cls_loss_history.extend(cls_loss_meter.history)
        
        train_loss = loss_meter.compute_average()
        train_box_loss = box_loss_meter.compute_average()
        train_cls_loss = cls_loss_meter.compute_average()
        
        train_loss_epochs.append(train_loss)
        train_box_loss_epochs.append(train_box_loss)
        train_cls_loss_epochs.append(train_cls_loss)

        if verbose:
            print('[train] epoch: {:>2d}, loss(box/cls) = {:.5f}({:.5f}/{:.5f}), time: {}' \
                .format(epoch+1, train_loss, train_box_loss, train_cls_loss, format_time(time.time() - t1)))

        # Validate
        t2 = time.time()     
        # v_loss_meter, v_score_meter = validate(model, device, valid_loader, criterion)

    if verbose:
        # print('[valid] best epoch {:>2d}, score = {:.5f}'.format(best_epoch+1, valid_best_score))
        print('training finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_history'     : train_loss_history,
        'train_box_loss_history' : train_box_loss_history,
        'train_cls_loss_history'  : train_cls_loss_history,

        'train_loss_epochs'     : train_loss_epochs,
        'train_box_loss_epochs' : train_box_loss_epochs,
        'train_cls_loss_epochs' : train_cls_loss_epochs,
    }
 
    return train_info

def run_loader(
    model,
    train_loader,
    valid_loader,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10,
    verbose=True
    ):

    if verbose:
        run_decription = (
            f"learning_rate = {learning_rate}\n"
            f"weight_decay = {weight_decay}\n"
            f"num_epoch = {num_epoch}\n"
        )
        print(run_decription)

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

def run(
    model,
    learning_rate=3e-4,
    weight_decay=1e-3,
    batch_size=8,
    num_epoch=10,
    verbose=True,
    debug=False
    ):

    # dataset = PennFudanDataset('../PennFudanPed', get_transform(train=False))

    dataset = PascalVOCDataset('./pascal-voc/', 'TRAIN', get_transform(train=True))
    # TODO:
    # valid_dataset = PascalVOCDataset('./pascal-voc/', 'VALID', get_transform(train=False))

    train_loader, valid_loader = create_dataloaders_sampler(dataset, dataset, batch_size=batch_size, debug=debug)

    train_info = run_loader(model, train_loader, valid_loader, learning_rate, weight_decay, num_epoch, verbose)
     
    return train_info

def main(debug=True):

    print('run main...')
    t0 = time.time()

    # SEED = 2020
    # seed_everything(SEED)
    # print_version()

    model = faster_rcnn()

    params = {
        'learning_rate' : 0.001,
        'weight_decay'  : 0,
        'batch_size'    : 2,
        'num_epoch'     : 2,
        'verbose'       : True,
        'debug'         : debug
    }

    train_info = run(model, **params)

    print('main finished for: {} '.format(format_time(time.time() - t0)))

    return train_info


if __name__ == "__main__":   
    main()