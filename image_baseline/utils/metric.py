import torch
import numpy as np
def MARKET_EVAL_FUNC(dismat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    '''
    :param dismat: 每个query与其对应查询结果的距离
    :param q_pids: 每个query行人的id
    :param g_pids: 每个查询结果的行人的id
    :param q_camids: 摄像头id，对于同一个query，如果查询结果与query都来自一个摄像头，则丢弃
    :param g_camids: 同上
    :param max_rank:
    :return:
    '''
    num_q, num_g = dismat.shape
    if num_g < max_rank:
        max_rank = num_g

    ##根据相似度进行排序
    indices = np.argsort(dismat, axis=1)

    ##在排序结果找到同id图片，用于计算cmc
    matchs = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        '''当前query对应查询结果的排序'''
        order = indices[q_idx]

        '''若查询结果与query同摄像机且同id，删除'''
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        '''计算CMC'''

        '''首先先筛选'''
        orig_cmc = matchs[q_idx][keep]

        '''全是false'''
        if not np.any(orig_cmc):
            continue

        '''累加和，因为orig_cmc里面是true，false，true为1，得到cmc列表'''
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        '''计算map'''

        '''预测正确的数量'''
        num_rel = orig_cmc.sum()

        '''累加和，用于计算'''
        tmp_cmc = orig_cmc.cumsum()

        '''计算准确率，前k个查询结果的准确率'''
        tmp_cmc = [x / (i + 1) for i, x in enumerate(tmp_cmc)]

        '''计算召回率，召回率每次变化都是当查询结果id等于query时，故乘orig_cmc'''
        '''最终的结果类似 0,0,1,0,0,2,0,0,3这种，每个有值的都是查询正确的'''
        '''然后后面除以查询结果中同id数量，也就是上面num_rel就是找召回率'''
        '''但是这里其实已经提前乘上准确率了，最终结果类似于 0,0,1/3,0,0,2/6,0,0,3/9'''
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc

        '''以上为例，除以同id数量后，假设为五个'''
        '''0，0，1/3 /5，0，0，2/6 /5，0，0，3/9 /5'''
        '''等价于 0,0, 1/3(准确率)*1/5(召回率) ....'''

        if num_rel == 0:
            AP = 0
            print('AP=0')
        else:
            AP = tmp_cmc.sum() / num_rel

        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    if len(all_AP) == 0:
        return 0, 0
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class MARKET_MAP():
    def __init__(self, num_query, max_rank=50, feat_norm=True,date=False,one_day=True):
        '''

        :param num_query: query数量
        :param max_rank:
        :param feat_norm:
        :param date:
        :param one_day:
        '''

        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.one_day = one_day
        self.date=date
        self.reset()

    def reset(self):
        '''
        feats:模型返回的特征
        pids: 图片中行人的id
        camids:图片摄像头的id
        :return:
        '''
        self.feats = []
        self.pids = []
        self.camids = []
        self.dates = []

    def update(self, output):


        self.feats.append(output[0])
        self.pids.append(output[1])
        self.camids.append(output[2])
        if len(output)==4:

           self.dates.append(date[3])


    def compute(self):
        feats = torch.stack(self.feats, dim=0)

        if self.feat_norm:
            feats = torch.nn.functional.normalize(feats, p=2)

        '''用于查询的图片的特征及其id和摄像头id'''

        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_dates = np.asarray(self.dates[:self.num_query])

        '''每个query查询结果的特征及其id和摄像头id'''
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_dates = np.asarray(self.dates[self.num_query:])
        if self.date:
            m, n = qf.shape[0], gf.shape[0]

            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()

            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()

            cmc, mAP = MARKET_EVAL_FUNC(distmat, q_pids, g_pids, q_camids, g_camids)
            return cmc,mAP

        else:
            if self.one_day:
                mean_map = 0
                date = 0
                q_index = np.where(q_dates == date)
                date_qf = qf[q_index]
                date_q_pids = q_pids[q_index]
                date_q_camids = q_camids[q_index]
                g_index = np.where(g_dates == date)
                date_gf = gf[g_index]
                date_g_pids = g_pids[g_index]
                date_g_camids = g_camids[g_index]
                m, n = date_qf.shape[0], date_gf.shape[0]
                if m == 0:
                    return 0
                distmat = torch.pow(date_qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(date_gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()

                distmat.addmm_(1, -2, date_qf, date_gf.t())
                distmat = distmat.cpu().numpy()

                cmc, mAP = MARKET_EVAL_FUNC(distmat, date_q_pids, date_g_pids, date_q_camids, date_g_camids)
                mean_map += mAP
                return cmc, mean_map
            else:
                mean_map = 0

                date = 0
                q_index = np.where(q_dates == date)

                date_qf = qf[q_index]
                date_q_pids = q_pids[q_index]
                date_q_camids = q_camids[q_index]

                g_index = np.where(g_dates != date)
                date_gf = gf[g_index]
                date_g_pids = g_pids[g_index]
                date_g_camids = g_camids[g_index]
                m, n = date_qf.shape[0], date_gf.shape[0]

                if m == 0 or n == 0:
                    return 0
                distmat = torch.pow(date_qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(date_gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()

                distmat.addmm_(1, -2, date_qf, date_gf.t())
                distmat = distmat.cpu().numpy()

                cmc, mAP = MARKET_EVAL_FUNC(distmat, date_q_pids, date_g_pids, date_q_camids, date_g_camids)
                mean_map += mAP
                return cmc, mean_map