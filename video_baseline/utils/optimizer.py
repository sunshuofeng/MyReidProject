from torchtools.optim import RangerLars
import torch
def build_optimizer(cfg,model):
    if cfg['optimizer']=='SGD':
        return torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['decay'])
    elif cfg['optimizer']=='Adam':
        return torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
    elif cfg['optimizer']=='RangerLars':
        return RangerLars(model.parameters(),lr=cfg['lr'])

