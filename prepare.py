import numpy as np
import string
import torch
import torch.nn.functional as F

def device():
    return 'cuda'if torch.cuda.is_available() else 'cpu'

def get_msa1hot(file):
    dic = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
           'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, '-': 20}
    # msa = np.array([])
    msa=[]
    for line in open(file, 'r').readlines():
        if line[0] != '>':
            seq = []
            for i in range(len(line) - 1):
                c = line[i]
                if c in dic:
                    seq.append(dic[c])
                elif c.isupper():
                    seq.append(dic['-'])  # 用'-'代替大写的，不在字典内的，不然长度不一致，小写则去掉
            msa.append(seq)
    msa=torch.from_numpy(np.array(msa)).long()
    msa1hot=F.one_hot(msa, 21).float()
    return msa1hot.to(device())


def get_MeffandWm(msa1hot, identity_rate=0.8):
    msa1hot # [193,140]
    total_score = []
    thred = identity_rate * msa1hot.shape[1] # 80%阈值
    for i in range(msa1hot.shape[0]):  # 193*140*21
        score = 0
        for j in range(msa1hot.shape[0]):
            score_temp = 0
            score_temp = torch.sum( msa1hot[i,:,:] * msa1hot[j,:,:])
            if score_temp > thred:
                score += 1
            else:
                score += 0
        total_score.append(score)
    score_array = np.array(total_score)
    score_array=torch.from_numpy(score_array).to(device())
    Wm = 1. / score_array
    Meff = torch.sum(Wm).to(device())
    return Meff, Wm

def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot) #
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w


def get_socalled_psfm(msa1hot, Meff, Wm):
    total_Wm = Wm[:, None, None] * msa1hot
    psfm = total_Wm.sum(0) / Meff + 1e-9  # 根据开源代码来的莫名公式
    return psfm.to(device())


def get_pe(psfm):
    pe = (-psfm * torch.log(psfm)).sum(dim=1)  # positional entropy 的计算公式
    return pe.to(device())


def get_feature_1d(msa1hot, Meff, Wm):
    seq1hot = msa1hot[0, :, :20].float()  # [140,20]要预测的序列的onehot（去除-）
    psfm = get_socalled_psfm(msa1hot, Meff, Wm)  # [140,21]
    pe = get_pe(psfm)  # [140]

    feature = torch.cat((seq1hot, psfm, pe[:, None]), dim=1)  # [140,42]
    feature = feature[None, :, :]  # [1,140,42]
    feature_h = feature[:, :, None, :].repeat(1, 1, msa1hot.shape[1], 1)  # [1,140,140,42]
    feature_v = feature[:, None, :, :].repeat(1, msa1hot.shape[1], 1, 1)  # [1,140,140,42]
    feature_total = torch.cat((feature_h, feature_v), dim=-1)  # [1,140,140,84]

    return feature_total.to(device())  # [1,140,140,84]



def get_feature_2d(msa1hot, Meff, Wm):
    x = msa1hot.view(msa1hot.shape[0], -1)  # [193,2940],不改变存储顺序
    num_points = Meff - torch.sqrt(Wm.mean())  # 不懂这在做什么
    mean = (x * Wm[:, None]).sum(dim=0, keepdims=True) / num_points  # 求中点？？
    x = (x - mean) * torch.sqrt(Wm[:, None])  # [193,2940] 去中心化再拉伸？？

    cov = (x.t() @ x) / num_points  # [2940,2940] 协方差矩阵 公式不一样？
    cov_reg = cov + 4.5 * torch.eye(msa1hot.shape[1] * msa1hot.shape[2]).to(device()) / torch.sqrt(Meff)
    cov_inv = torch.inverse(cov_reg)  # 精度矩阵

    feature = cov_inv.view(msa1hot.shape[1], msa1hot.shape[2], msa1hot.shape[1], msa1hot.shape[2])
    S_reg = feature[:, :20, :, :20]
    feature = feature.transpose(1, 2).contiguous().reshape(msa1hot.shape[1], msa1hot.shape[1], -1)
    # [140,140,441]
    S_reg = S_reg**2
    S_hat = torch.sqrt(S_reg.sum(dim=(1, 3))) * (1 - torch.eye(msa1hot.shape[1]).to(device()))  # [140,140],除去对角线
    APC = S_hat.sum(dim=0, keepdims=True) * S_hat.sum(dim=1, keepdims=True) / S_hat.sum()  # [140，140]
    contacts = (S_hat - APC) * (1 - torch.eye(msa1hot.shape[1]).to(device()))  # [140,140]毁灭吧

    feature = torch.cat((feature, contacts[:, :, None]), dim=2)  # [140,140,442]
    return feature[None, :, :, :].to(device())  # [1,140,140,442]


def feature_deal(msa_file):
    msa1hot = get_msa1hot(msa_file)
    Meff, Wm = get_MeffandWm(msa1hot, 0.8)
    feature_1d = get_feature_1d(msa1hot, Meff, Wm)  # [1,140,140,84]
    feature_2d = get_feature_2d(msa1hot, Meff, Wm)  # [1,140,140,442]
    feature_total = torch.cat((feature_1d, feature_2d), dim=3)  # [1,140,140,526]
    feature_total = feature_total.permute((0, 3, 2, 1))  # [1,526,140,140]

    return feature_total

