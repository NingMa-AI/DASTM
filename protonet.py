import time

import torch.nn as nn
import torch
import numpy as np
from mmskl.st_gcn_aaai18 import ST_GCN_18
from utils import get_support_query_data, extract_k_segement, compute_similarity, euclidean_dist, euclidean_distance
from torch.nn import functional as F
import gl
from soft_dtw import SoftDTW
from cross_attention import CrossAttention

class ProtoNet(nn.Module):

    def __init__(self, opt):
        super(ProtoNet, self).__init__()

        if 'ntu' in gl.dataset:
            node = 25
            ms_graph = 'graph.ntu_rgb_d.AdjMatrixGraph'
            sh_grpah = 'shift_gcn_graph.ntu_rgb_d.Graph'
            st_graph = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        elif gl.dataset == 'kinetics':
            node = 18
            ms_graph = 'graph.kinetics.AdjMatrixGraph'
            sh_grpah = 'shift_gcn_graph.kinetics.Graph'
            st_graph = {'layout': 'openpose', 'strategy': 'spatial'}
        else:
            ms_graph = None
            sh_grpah = None
            st_graph = None
            node = 0

        self.model = ST_GCN_18(
            in_channels=3,
            num_class=60,
            dropout=0.1,
            edge_importance_weighting=False,
            graph_cfg=st_graph
        )
        self.out_channel = 256

        if gl.SA == 1:
            self.attention_x = CrossAttention(num_attention_heads=1, input_size=self.out_channel, hidden_size=self.out_channel, hidden_dropout_prob=0.2)
            self.attention_y = CrossAttention(num_attention_heads=1, input_size=self.out_channel, hidden_size=self.out_channel, hidden_dropout_prob=0.2)
        else:
            self.attention_x = None
            self.attention_y = None

    def loss(self, input, target, n_support, dtw):
        # input is encoder by ST_GCN
        n, c, t, v = input.size()

        def supp_idxs(cc):
            # FIXME when torch will support where as np
        
            return torch.nonzero(target.eq(cc))[:n_support].squeeze(1)

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target)
        n_class = len(classes)
        # FIXME when torch will support where as np
        # assuming n_query, n_target constants
        n_query = target.eq(classes[0].item()).sum().item() - n_support

        support_idxs = list(map(supp_idxs, classes))
        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, c, t, v)

        # FIXME when torch will support where as np
        query_idxs = torch.stack(list(map(lambda c: torch.nonzero(target.eq(c))[n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]

        z_proto = z_proto.view(n_class, n_support, c, t, v).mean(1)  # n, c, t, v

        if dtw > 0:
            dist, reg_loss = self.dtw_loss(zq, z_proto)
        else:
            zq = zq.view(n_class * n_query, -1)
            z_proto = z_proto.view(n_class, -1)
            dist = euclidean_dist(zq, z_proto)
            reg_loss = torch.tensor(0).float().to(gl.device)

        log_p_y = F.log_softmax(-dist, dim=1).view(n_class, n_query, -1)

        target_inds = torch.arange(0, n_class).to(gl.device)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        if gl.reg_rate > 0:
            loss_val += reg_loss

        return loss_val, acc_val, reg_loss

    def dtw_loss(self, zq, z_proto):
        if self.attention_x != None:
            zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
            z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
            dist = self.attention_dtw_dist(zq, z_proto)
        else:
            z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
            zq = zq.permute(0, 2, 3, 1).contiguous()
            dist = self.dtw_dist(zq, z_proto)

        reg_loss = torch.tensor(0).float().to(gl.device)

        if gl.reg_rate > 0:
            reg_loss = self.svd_reg_spatial(z_proto) + self.svd_reg_spatial(zq)
            rate = gl.reg_rate
            reg_loss = reg_loss * rate

        return dist, reg_loss


    def attention_dtw_dist(self, x, y):
        '''
            :param x: [n, t, c] z_query
            :param y: [m, t, c] z_proto
            :return: [n, m]
        '''

        n, t, v, c = x.size()
        m, _, _, _ = y.size()

        x = x.unsqueeze(1).expand(n, m, t, v, c).reshape(n * m, t, v, c)
        y = y.unsqueeze(0).expand(n, m, t, v, c).reshape(n * m, t, v, c)

        sdtw = SoftDTW(gamma=gl.gamma, normalize=False, attention=self.attention_x, attention_y=self.attention_y)
        loss = sdtw(x, y)

        return loss.view(n, m)

    def dtw_dist(self, x, y):

        if len(x.size()) == 4:
            n, t, v, c = x.size()
            x = x.view(n, t, v * c)
            y = y.view(-1, t, v * c)

        n, t, c = x.size()
        m, _, _ = y.size()

        x = x.unsqueeze(1).expand(n, m, t, c).reshape(n * m, t, c)
        y = y.unsqueeze(0).expand(n, m, t, c).reshape(n * m, t, c)
        sdtw = SoftDTW(gamma=gl.gamma, normalize=False, attention=self.attention_x, attention_y=self.attention_y)
        loss = sdtw(x, y)

        return loss.view(n, m)

    def svd_reg_spatial(self, x):

        if len(x.size()) == 4:

            n, t, v, c = x.size()
            x = x.view(-1,v,c)

        loss = torch.tensor(0).float().to(gl.device)

        for i in range(x.size()[0]):

            transpose_X = x[i]

            # fast version
            softmax_tgt = torch.softmax((transpose_X - torch.max(transpose_X)), dim=1)
            list_svd, _ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_tgt, 2), dim=0)), descending=True)
            method_loss = -torch.mean(list_svd[:min(softmax_tgt.shape[0], softmax_tgt.shape[1])])
            loss += method_loss

        return loss / x.size()[0]

   
    def forward(self, x):
        x = self.model(x)

        return x

