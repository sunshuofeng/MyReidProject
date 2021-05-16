import torch
import math
from bisect import bisect_right
from torchtools.lr_scheduler import DelayedCosineAnnealingLR

class Delay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, last_epoch=-1):

        self.epochs = epochs
        super(Delay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < int(self.epochs * 0.3):
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.last_epoch - int(self.epochs * 0.3)) / (self.epochs - int(self.epochs * 0.3)) for base_lr in self.base_lrs]


class Flat(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, cfg,start_lr,anneal_start=0.6,t=4, last_epoch=-1):
        self.epochs = cfg['epochs']
        self.start = anneal_start
        self.t=t
        self.start_lr=start_lr


        super(Flat, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < int(self.epochs * self.start):
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [
                1e-5+(self.start_lr-1e-5)*(1+math.cos(math.pi*(self.last_epoch-int(self.epochs*self.start))/t))/2
                for base_lr in self.base_lrs
            ]


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones=[30,55],
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def build_scheuler(cfg,optimizer,epochs,milestones=None):
    if cfg['scheduler']=='Delay':
        return Delay(optimizer,epochs)
    elif cfg['scheduler']=='CosDelay':
        return DelayedCosineAnnealingLR(optimizer,int(0.3*epochs),5)
    elif cfg['scheduler']=='Flat':
        return Flat(optimizer,epochs)
    elif cfg['scheduler']=='Warmup':
        return WarmupMultiStepLR(optimizer,milestones)



