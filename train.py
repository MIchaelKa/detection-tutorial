import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from faster_rcnn import faster_rcnn
from utils import format_time, seed_everything
from utils import calculate_mAP
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
        all_number = 100

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
        print(f'data loader size, train: {len(train_loader)}, valid: {len(valid_loader)}\nbatch_size = {batch_size}')
    
    return train_loader, valid_loader

def train_epoch(model, device, criterion, train_loader, optimizer, verbose=True):
    
    t0 = time.time()
    print_every = 10

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

def validate(model, device, criterion, valid_loader, verbose=True):

    t0 = time.time()
    print_every = 100

    model.eval()
    
    # loss
    loss_meter = AverageMeter()
    box_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()

    # mAP
    all_pred_boxes = []
    all_pred_conf = []
    all_true_boxes = []
    all_true_labels = []

    with torch.no_grad():
        for index, (image_batch, target_batch) in enumerate(valid_loader):

            image_batch = torch.stack(image_batch, dim=0)   
            image_batch = image_batch.to(device)
            
            offsets, labels = model(image_batch)

            loss, box_loss, cls_loss = criterion(labels, offsets, target_batch)
            
            # Update meters
            loss_meter.update(loss.item())
            box_loss_meter.update(box_loss.item())
            cls_loss_meter.update(cls_loss.item())

            # Detection
            pred_boxes, pred_conf = model.detect(offsets, labels, prob_threshold=0.5, max_overlap=0.7)

            # Save for mAP calculation
            true_boxes = [t['boxes'] for t in target_batch]
            true_labels = [t['labels'] for t in target_batch]

            all_pred_boxes.extend(pred_boxes)
            all_pred_conf.extend(pred_conf)
            all_true_boxes.extend(true_boxes)
            all_true_labels.extend(true_labels)

            if verbose and index % print_every == 0:
                print('[valid] index: {:>2d}, loss(box/cls) = {:.5f}({:.5f}/{:.5f}) time: {}' \
                    .format(
                        index,
                        loss_meter.compute_average(),
                        box_loss_meter.compute_average(),
                        cls_loss_meter.compute_average(),
                        format_time(time.time() - t0)
                    )
                )
    if verbose:
        print('[valid] calculate_mAP... time: {}'.format(format_time(time.time() - t0)))
    
    mAP = calculate_mAP(all_pred_boxes, all_pred_conf, all_true_boxes, all_true_labels, device, threshold=0.5)

    if verbose:
        print('[valid] mAP = {:.5f},  time: {}'.format(mAP, format_time(time.time() - t0)))
   
    return (loss_meter, box_loss_meter, cls_loss_meter), mAP


def train_model(model, device, criterion, train_loader, valid_loader, optimizer, num_epochs, verbose=True):
       
    t0 = time.time()

    train_loss_history = []
    train_box_loss_history = []
    train_cls_loss_history = []
    
    train_loss_epochs = []
    train_box_loss_epochs = []
    train_cls_loss_epochs = []

    valid_loss_epochs = []
    valid_box_loss_epochs = []
    valid_cls_loss_epochs = []
    valid_scores = []

    if verbose:
        print('training started...')
    
    for epoch in range(num_epochs):

        # Train
        t1 = time.time()
        loss_meters = train_epoch(model, device, criterion, train_loader, optimizer, verbose=False)

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
        loss_meters, score = validate(model, device, criterion, valid_loader, verbose=False)

        loss_meter, box_loss_meter, cls_loss_meter = loss_meters

        valid_loss = loss_meter.compute_average()
        valid_box_loss = box_loss_meter.compute_average()
        valid_cls_loss = cls_loss_meter.compute_average()
        
        valid_loss_epochs.append(valid_loss)
        valid_box_loss_epochs.append(valid_box_loss)
        valid_cls_loss_epochs.append(valid_cls_loss)
        valid_scores.append(score)

        if verbose:
            print('[valid] epoch: {:>2d}, loss(box/cls) = {:.5f}({:.5f}/{:.5f}), mAP = {:.5f},  time: {}' \
                .format(epoch+1, valid_loss, valid_box_loss, valid_cls_loss, score, format_time(time.time() - t1)))

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

        'valid_loss_epochs'     : valid_loss_epochs,
        'valid_box_loss_epochs' : valid_box_loss_epochs,
        'valid_cls_loss_epochs' : valid_cls_loss_epochs,

        'valid_scores' : valid_scores,
    }
 
    return train_info

def run_loader(
    generate_anchors_settings,
    box_loss_settings,
    train_loader,
    valid_loader,
    learning_rate=3e-4,
    weight_decay=1e-3,
    num_epoch=10,
    verbose=True,
    print_params=True,
    ):

    if print_params:  
        run_decription = (
            f"learning_rate = {learning_rate}\n"
            f"weight_decay = {weight_decay}\n"
            f"num_epoch = {num_epoch}\n"
        )
        print(run_decription)

        run_decription = (
            f"generate anchors settings:\n"
            f"feature_dims = {generate_anchors_settings['feature_dims']}\n"
            f"feature_map_scales = {generate_anchors_settings['feature_map_scales']}\n"
            f"aspect_ratios = {generate_anchors_settings['aspect_ratios']}\n"
            f"clip = {generate_anchors_settings['clip']}\n"
        )
        print(run_decription)

        run_decription = (
            f"box loss settings:\n"
            f"anchor_threshold = {box_loss_settings['anchor_threshold']}\n"
            f"fix_no_anchors = {box_loss_settings['fix_no_anchors']}\n"
            f"HNM settings:\n"
            f"enable_hnm = {box_loss_settings['enable_hnm']}\n"
            f"neg_pos_ratio = {box_loss_settings['neg_pos_ratio']}\n"
        )
        print(run_decription)

    device = get_device()

    model = faster_rcnn(device, generate_anchors_settings).to(device)
    criterion = BoxLoss(device, box_loss_settings, model.anchors)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )

    train_info = train_model(model, device, criterion, train_loader, valid_loader, optimizer, num_epoch)
    
    return train_info, model

def run(
    generate_anchors_settings,
    box_loss_settings,
    learning_rate=3e-4,
    weight_decay=1e-3,
    batch_size=8,
    num_epoch=10,
    verbose=True,
    print_params=True,
    debug=False
    ):

    # dataset = PennFudanDataset('../PennFudanPed', get_transform(train=False))

    dataset = PascalVOCDataset('./pascal-voc/', 'TRAIN', get_transform(train=True))
    # TODO:
    # valid_dataset = PascalVOCDataset('./pascal-voc/', 'VALID', get_transform(train=False))

    train_loader, valid_loader = create_dataloaders_sampler(dataset, dataset, batch_size=batch_size, debug=debug)

    train_info, model = run_loader(
        generate_anchors_settings,
        box_loss_settings,
        train_loader,
        valid_loader,
        learning_rate,
        weight_decay,
        num_epoch,
        verbose,
        print_params
    )
     
    return train_info, valid_loader, model

def main():

    print('run main...')
    t0 = time.time()

    SEED = 2021
    seed_everything(SEED)

    generate_anchors_settings = dict(
        clip=False,
        feature_dims = [25, 13, 7, 4], # TODO: Read from image?
        # scales = [0.9, 0.6, 0.3],
        feature_map_scales = [0.2, 0.4, 0.6, 0.8],
        aspect_ratios = [1., 2., 0.5],
    )

    box_loss_settings = dict(
        anchor_threshold = 0.3,
        fix_no_anchors = False,

        # Hard Negative Mining settings
        enable_hnm = False,
        neg_pos_ratio = 3.0
    )

    backbone_settings = dict(
        name = 'resnet18' # 'resnext50' 'effnet'
    )

    detection_settings = dict(
        clip_predictions=False,
        prob_threshold=0.5,
        max_overlap=0.5
    )

    evaluation_settings = dict(
        mAP_threshold=0.5,
    )
    
    # compare with trainimg scheduler_params impl!
    # scheduler_params = dict(
    #     mode='max',
    #     factor=0.7,
    #     patience=0,
    #     verbose=False, 
    #     threshold=0.0001,
    #     threshold_mode='abs',
    #     cooldown=0, 
    #     min_lr=1e-8,
    #     eps=1e-08
    # )

    params = {
        'generate_anchors_settings' : generate_anchors_settings,
        'box_loss_settings' : box_loss_settings,

        # 'rpn_heads'         : 0,

        'learning_rate'     : 0.001,
        'weight_decay'      : 0,
        'batch_size'        : 8,
        'num_epoch'         : 5,
        'verbose'           : True,
        'print_params'      : True, # False in notebook
        'debug'             : False
    }

    train_info = run(**params)

    print('main finished for: {} '.format(format_time(time.time() - t0)))

    return train_info


if __name__ == "__main__":   
    main()