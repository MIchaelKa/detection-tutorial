import numpy as np

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.total_sum = 0
        self.total_count = 0

    def update(self, x):
        self.history.append(x)
        self.total_sum += x
        self.total_count += 1

    def compute_average(self):
        return np.mean(self.history)

    def moving_average(self, alpha):
        avg_history = [self.history[0]]
        for i in range(1, len(self.history)):
            moving_avg = alpha * avg_history[-1] + (1 - alpha) * self.history[i]
            avg_history.append(moving_avg)
        return avg_history