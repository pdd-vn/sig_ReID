# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
import torch.nn as nn


__all__ = ["pairwise_circleloss", "pairwise_cosface"]


def pairwise_circleloss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float=0.25,
        gamma: float=128, ) -> torch.Tensor:
    # Normalize embedding vector
    embedding = F.normalize(embedding, dim=1)

    # Calculate the similarity matrix
    similarity_matrix = torch.matmul(embedding, embedding.t())

    # Number of input vector
    N = similarity_matrix.size(0)

    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    is_pos = is_pos.bool()
    is_neg = is_neg.bool()
    
    similarity_matrix = similarity_matrix.view(-1)
    
    # Get the upper triangular part of matrix
    is_pos = is_pos.triu(diagonal=1).view(-1)
    is_neg = is_neg.triu(diagonal=1).view(-1)

    # Calculate the similarity of the positive pairs and the negative pairs
    s_p = similarity_matrix[is_pos]
    s_n = similarity_matrix[is_neg]

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - gamma * alpha_p * (s_p - delta_p)
    logit_n = gamma * alpha_n * (s_n - delta_n)  

    loss = F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)).mean()
    # print("Circle loss is {}".format(loss))
    # import ipdb; ipdb.set_trace()
    return loss


def pairwise_circleloss_wrong(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float=0.25,
        gamma: float=128, ) -> torch.Tensor:

    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)

    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - gamma * alpha_p * (s_p - delta_p) + (-99999999.) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    print("Circle loss is {}".format(loss))
    # import ipdb; ipdb.set_trace()
    return loss


def pairwise_circleloss_forgery(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        real_forg: torch.Tensor,
        margin: float=0.25,
        gamma: float=128, ) -> torch.Tensor:
    
    '''
    sim
        A1  A2  B1  B2
    A1  1   1   0   0
    A2  1   1   0   0
    B1  0   0   1   1
    B2  0   0   1   1

    forg
        A1  A2  B1  B2   
        0.5 1   0.5 1
        0.5 1   0.5 1
        0.5 1   0.5 1
        0.5 1   0.5 1

    '''
    # Normalize embedding vector
    embedding = F.normalize(embedding, dim=1)

    # Calculate the similarity matrix
    similarity_matrix = torch.matmul(embedding, embedding.t())

    # Number of input vector
    N = similarity_matrix.size(0)

    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    is_pos = is_pos.bool()
    is_neg = is_neg.bool()
    
    similarity_matrix = similarity_matrix.view(-1)
    
    # Get the upper triangular part of matrix
    is_pos = is_pos.triu(diagonal=1).view(-1)
    is_neg = is_neg.triu(diagonal=1).view(-1)

    # Calculate the similarity of the positive pairs and the negative pairs
    s_p = similarity_matrix[is_pos]
    s_n = similarity_matrix[is_neg]


    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - gamma * alpha_p * (s_p - delta_p)
    logit_n = gamma * alpha_n * (s_n - delta_n)  

    loss = F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)).mean()

    return loss



def pairwise_cosface(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
    # Normalize embedding features
    embedding = F.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
    logit_n = gamma * (s_n + margin) + (-99999999.) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    # def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        sp, sn = convert_label_to_similarity(pred, label)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss