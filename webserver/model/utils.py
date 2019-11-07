import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
import torch.nn.functional as F


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cal_l2(x, y):
    return torch.pow((x - y), 2).sum(-1).sum()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, f_dic, B, N, size_average=True):
        
        out = torch.zeros(1).to(self.device)
        # Postive Samples Within Group Loss
        # Assume the size of each feature is (B x N) 
        for kk in f_dic.keys():
#             pdb.set_trace()
            mat = f_dic[kk]
            L = mat.size(0)
            if L != 1:
                mat_dup = mat.unsqueeze(0).expand(L, L, N)
                batch_dup = mat.unsqueeze(1).expand(L, L, N)
                distances = (mat_dup - batch_dup).pow(2).sum(dim=-1).sum()
                out += (0.5 * distances / 6)
        
        if len(f_dic) == 1:
            pass
        else:
            for k1, k2 in list(combinations(f_dic, 2)):
                b1 = len(f_dic[k1])
                b2 = len(f_dic[k2])
                for bb in range(b2):
#                     pdb.set_trace()
                    distances = cal_l2(f_dic[k1], f_dic[k2][bb].unsqueeze(0).expand(b1, N))/(b1+b2)
                    out += (0.5 * F.relu(self.margin - (distances + self.eps)).pow(2))

        return out
