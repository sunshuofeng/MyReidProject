from torch.cuda.amp import autocast,GradScaler
scaler=GradScaler()
from tqdm import tqdm
import torch
from utils.metric import  *
class Trainer:
    def __init__(self,train_cfg):
        self.train_loaders = train_cfg['train_loaders']
        self.val_loader = train_cfg['val_loader']
        self.logger = train_cfg['logger']
        self.num_class = train_cfg['num_class']
        self.num_query = train_cfg['num_query']
        self.checkpoint = train_cfg['checkpoint']
        self.model=train_cfg['model']
        self.optimizer=train_cfg['optimizer']
        self.scheduler=train_cfg['scheduler']
        self.criterion=train_cfg['criterion']


    def train_one_epoch(self,device,loader):
        for images,pids in loader:
            images=images.to(device)
            pids=pids.to(device)
            self.optimizer.zero_grad()
            with autocast():
                out=self.model(images)
                if 'final_loss' in out.keys():
                    loss=out['final_loss']
                else:
                    loss=self.criterion(out,pids)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

    def train(self,cfg):
        epochs=cfg['epochs']
        max_map=0
        device=torch.device('cuda:0')
        with tqdm(total=epochs*len(self.train_loaders)) as pbar:
            for epoch in range(epochs):
                for loader in self.train_loaders:
                    self.model.train()
                    self.train_one_epoch(device,loader)
                    self.scheduler.step()
                    self.model.eval()
                    if epoch%5==0:
                        rank1,map=self.do_eval(device,cfg['date'])
                        print('epoch:{}---map:{}---rank1:{}'.format(epoch,map,rank1))
                        self.logger.info('epoch:{}---map:{}'.format(epoch,map))
                        if map>max_map:
                            max_map=map
                            torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                                       'result.pt')
                    pbar.update(1)

    def do_eval(self,device,date):

        metric=MARKET_MAP(self.num_query,date=True)
        with torch.no_grad():
            
            for images,pids,cams,dates in self.val_loader:
                images=images.to(device)
                outs=self.model(images)
                for out in zip(outs['feature'],pids,cams):
     
         cmc,map=metric.compute()
         return map



