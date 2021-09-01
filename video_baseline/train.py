from torch.cuda.amp import autocast,GradScaler
scaler=GradScaler()
from tqdm import tqdm
import torch
from utils.metric import  *
class Trainer:
    def __init__(self,train_cfg):
        self.train_loader= train_cfg['train_loader']
        self.val_loader = train_cfg['val_loader']
        self.logger = train_cfg['logger']
        self.num_class = train_cfg['num_class']
        self.checkpoint = train_cfg['checkpoint']
        self.model=train_cfg['model']
        self.optimizer=train_cfg['optimizer']
        self.scheduler=train_cfg['scheduler']
        self.criterion=train_cfg['criterion']



    def train_one_epoch(self,device):
        for images,pids in self.train_loader:
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
        epochs=['epochs']
        max_map=0
        device=torch.device('cuda:0')
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                    self.model.train()
                    self.train_one_epoch(device)
                    self.scheduler.step()
                    self.model.eval()
                    map=self.do_eval()
                    print('epoch:{}----map:{}'.format(epoch,map))
                    if map>max_map:
                        max_map=map
                        torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict()},'result.pt')
                    pbar.update(1)

    def do_eval(self):
        gallery_features = []
        gallery_labels = []
        gallery_cams = []
        with torch.no_grad():
            for images, labels, cams in self.val_loader:
                B, C, H, W = images.shape
                images = images.reshape(B // 6, 6, C, H, W)
                images = images.float().cuda()
                out = self.model(images)
                gallery_features.append(out['feature'].detach().cpu())
                gallery_labels.append(labels)
                gallery_cams.append(cams)
        gallery_features = torch.cat(gallery_features, dim=0).numpy()
        gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
        gallery_cams = torch.cat(gallery_cams, dim=0).numpy()

        _, mAP = Video_Cmc(gallery_features, gallery_labels, gallery_cams, self.val_loader.dataset.query_idx, 10000)
        return mAP




