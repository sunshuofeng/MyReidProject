
from datasets.dataset import *
from datasets.sampler import *
from datasets.transforms import *
import torch
import torch.utils.data as Data

data_root={
    'Market1501':'data/market',
    'VC':'data/vc',
    'P-DESTRE':'data/Pdata',
    'Real28':'data/real28',
    'Duke':'data/duke'
}
dataset_type={
    'Market1501':Market1501,
    'VC':VC_Clothes,
    'P-DESTRE':PDataset,
    'Real28':Real28,
    'Duke':Duke

}


def train_collate_fn(batch):
    imgs, pids, camid,path,date = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

###对于验证集而言，为了提高验证的真实性，我们应该防止同一摄像头的图片进入验证（同一摄像头相当于数据泄露）
def val_collate_fn_date(batch):
    imgs, pids, camids, path,date= zip(*batch)
    return torch.stack(imgs, dim=0),pids, camids,date


def make_dataloader(dataset, cfg, pid_add=0):
    train_transform = build_transforms(cfg, training=True)
    test_transform = build_transforms(cfg, training=False)
    data = dataset
    num_classes = data.num_train_pids
    train_set=ImageDataset(data.train,train_transform,pid_add)
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



def MAKE_DATALOADER(cfg):
    train_names = cfg['train_data']
    val_name = cfg['val_data']
    val_loader_dict = {}
    train_loaders = []
    all_num_class = 0
    for name in train_names:
        root = data_root[name]
        dataset=dataset_type[name](root)
        train_loader, val_loader, num_query, num_classes = make_dataloader(dataset, cfg, all_num_class)
        train_loaders.append(train_loader)
        val_loader_dict[name] = [val_loader, num_query]
        all_num_class += num_classes
    val_loader, num_query = val_loader_dict[val_name]
    return train_loaders, val_loader, num_query, all_num_class
