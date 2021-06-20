import numpy as np

import torch

from faster_rcnn import faster_rcnn
from utils import format_time

import time



def main():
    print('main')

    t1 = time.time()

    net = faster_rcnn()

    x = torch.rand(50, 3, 300, 300)

    out = net(x)

    print('time: {} '.format(format_time(time.time() - t1)))


if __name__ == "__main__":   
    main()