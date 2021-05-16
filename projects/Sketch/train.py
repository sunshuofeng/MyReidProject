from torch.cuda.amp import autocast,GradScaler

from torchtools.optim import RangerLars
import torch
from tqdm import tqdm
import os
import sys
from utils.metric import *
scaler=GradScaler()
class Trainer:
    def __init__(self,train_cfg):
        self.train_loader = train_cfg['train_loader']
        self.val_loader = train_cfg['val_loader']
        self.logger = train_cfg['logger']
        self.num_class = train_cfg['num_class']
        self.num_query = train_cfg['num_query']
        self.model=train_cfg['model']

    def train_one_epoch(self,device):
        for images,sketchs,pids in self.train_loader:
            self.optimizer.step()
            sketchs=sketchs.to(device)
            pids=pids.to(device)
            with autocast():
                _,loss=self.model(sketchs,pids)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()



    def train(self,cfg):
        max_map=0
        device=torch.device('cuda:0')
        self.model=self.model.to(device)
        step1_epochs=cfg['step1_epochs']
        self.model.lambdak.requires_grad=False
        ##rangerlars may be better than sgd in step1
        self.optimizer=RangerLars(self.model.parameters(),lr=1e-3)
        with tqdm(total=step1_epochs) as pbar:
            for epoch in range(step1_epochs):
                self.model.train()
                self.train_one_epoch(device)
                self.model.eval()

                map0,map1=self.do_eval(device)
                self.logger.info('step1:--epoch:{}--map0:{}---map1:{}'.format(epoch,map0,map1))
                if (map0+map1)/2>max_map:
                    max_map=(map0+map1)/2
                    torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict()},'step1.pt')
                pbar.update(1)
        step2_epochs=cfg['step2_epochs']
        self.model.lambdak.requires_grad=True
        lambda_params = []
        other_params=[]
        for pname, p in self.model.named_parameters():
                if pname=='lambdak':
                   lambda_params+=[p]
                else:
                    other_params+=[p]
        self.optimizer = torch.optim.SGD([
                    {'param':other_params,'lr':0.001,'momentum':0.9,'weight_decay':1e-4},
                    {'param':lambda_params,'lr':0.0001,'momentum':0.9,'weight_decay':1e-4}
                ])
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, 0.1)
        with tqdm(total=step2_epochs) as pbar:
            self.model.train()
            self.train_one_epoch(device)
            scheduler.step()

            self.model.eval()

            map0, map1 = self.do_eval(device)
            self.logger.info('step2:--epoch:{}--map0:{}---map1:{}'.format(epoch, map0, map1))
            if (map0 + map1) / 2 > max_map:
                max_map = (map0 + map1) / 2
                torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'step2.pt')

            pbar.update(1)
        step3_epochs=cfg['step3_epochs']
        self.model.lambdak.requires_grad=False
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=0.0001,momentum=0.9,weight_decay=1e-4)
        with tqdm(total=step3_epochs) as pbar:
            self.model.train()
            self.train_one_epoch(device)

            self.model.eval()

            map0, map1 = self.do_eval(device)
            self.logger.info('step3:--epoch:{}--map0:{}---map1:{}'.format(epoch, map0, map1))
            if (map0 + map1) / 2 > max_map:
                    max_map = (map0 + map1) / 2
                    torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'step3.pt')

            pbar.update(1)
    def do_eval(self,device):
        metric0=MARKET_MAP(self.num_query,date=False,one_day=True)
        metric1 = MARKET_MAP(self.num_query,date=False,one_day=False)
        with torch.no_grad():
            for imgs, sketchs, pids, cams, dates in self.val_loader:

                sketchs = sketchs.to(device)
                feat, _ = self.reid_model(sketchs, pids)
                for out in zip(feat, pids, cams, dates):
                    metric0.update(out)
                    metric1.update(out)
        cmc0,map0 = metric0.compute()
        cmc1, map1 = metric1.compute()
        return map0, map1
