
from datasets.dataset import *
from datasets.sampler import *
from datasets.transforms import *
import torch
import torch.utils.data as Data



def train_collate_fn(batch):
    imgs, pids, camid,path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

###对于验证集而言，为了提高验证的真实性，我们应该防止同一摄像头的图片进入验证（同一摄像头相当于数据泄露）
def val_collate_fn_date(batch):
    imgs, pids, camids, path= zip(*batch)
    return torch.stack(imgs, dim=0),pids, camids


def make_dataloader(cfg):
    dataset=Market1501(cfg['root'])
    train_transform = build_transforms(cfg, training=True)
    test_transform = build_transforms(cfg, training=False)
    data = dataset
    num_classes = data.num_train_pids
    train_set=ImageDataset(data.train,train_transform)
    val_set = ImageDataset(data.query + data.gallery, test_transform)
    val_collate = val_collate_fn_date
    train_loader = Data.DataLoader(train_set, batch_size=cfg['train_bs'],
                                       sampler=DateRandomIdentitySampler(data.train,
                                                                         cfg['train_bs'], cfg['train_K_instances']),
                                       num_workers=cfg['num_workers'],
                                       collate_fn=train_collate_fn)
    val_loader = Data.DataLoader(
        val_set, batch_size=cfg['test_bs'], shuffle=False, num_workers=cfg['num_workers'],
        collate_fn=val_collate
    )
    return train_loader, val_loader, len(data.query), num_classes



