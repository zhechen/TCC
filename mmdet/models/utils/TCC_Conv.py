import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from mmcv.cnn import build_conv_layer
from mmdet.models.plugins import DropBlock

class TCC_Conv(nn.Module):
    def __init__(self, in_ch, num_c=32, num_k=3, num_layers=3, tr_num_c=8): 
        super(TCC_Conv, self).__init__()
        self.in_ch = in_ch
        self.num_k = num_k
        #self.num_g = num_g
        self.num_c = num_c 
        nhead = 1
        self.tr_num_c = nhead * tr_num_c 
        self.num_layers = num_layers 
        

        in_convs = []
        in_k_convs = [];
        gk_convs = []
        in_q_convs = [];
        o_convs = []
        pos_embed_convs = []
        inc_norms = []
        for li in range(self.num_layers):
            dcn=dict(type='DCNv2', deform_groups=1)
            in_convs.append(nn.Conv2d(self.in_ch, self.num_c, 1, 1, padding=0)) 

            in_q_conv = build_conv_layer(dcn, self.num_c, self.tr_num_c,
                                         kernel_size=3,
                                         stride=1,
                                         padding=(2*li+3),
                                         dilation=(2*li+3))  
            in_q_convs.append(in_q_conv) 

            in_k_conv = build_conv_layer(dcn, self.num_c, self.tr_num_c,
                                         kernel_size=3,
                                         stride=1, padding=1) 
            in_k_convs.append(in_k_conv)

            gk_convs.append(nn.Conv2d(self.tr_num_c, self.num_k , 1, 1, padding=0)) 

            pos_embed_convs.append(nn.Conv2d(self.num_k*2, self.num_k * self.tr_num_c, 1, 1, padding=0))

            inc_norms.append(nn.InstanceNorm2d(self.tr_num_c))

            o_convs.append(nn.Conv2d(3 * self.tr_num_c + self.num_k, in_ch, 1, 1, padding=0)) 

        self.in_convs = nn.ModuleList(in_convs)
        self.in_q_convs = nn.ModuleList(in_q_convs)
        self.gk_convs = nn.ModuleList(gk_convs)
        self.pos_embed_convs = nn.ModuleList(pos_embed_convs)
        self.in_k_convs = nn.ModuleList(in_k_convs)
        self.inc_norms = nn.ModuleList(inc_norms)
        self.o_convs = nn.ModuleList(o_convs)
        
        self.dim_factor = 1. / float(math.sqrt(self.tr_num_c))
        self.dropout_o = nn.Dropout(p=0.1)

    def forward(self,x): 
        nx = x.size(0)
        cc = x.size(1)
        wx = x.size(3) # W'
        hx = x.size(2) # H'

        xind = torch.arange(wx, device=x.device).view(1, 1, 1, 1, wx)
        yind = torch.arange(hx, device=x.device).view(1, 1, 1, hx, 1)

        for li in range(self.num_layers):
            xi = F.relu(self.in_convs[li](x))

            # find and collect local features
            xquery = self.in_q_convs[li](xi) # N TC H W
            xquery_ = xquery.unsqueeze(2) # N TC 1 H W  

            # predict, find, and collect K global context features
            gkv = self.in_k_convs[li](xi) # N TC H W
            gk_mask = self.gk_convs[li](F.relu(gkv)) # N K H W
            gk_mask_ = gk_mask.view(nx, self.num_k, -1) # N K HW
            max_scr, max_inds = torch.max(gk_mask_, dim=2) # N K 
            with torch.no_grad():
                h_max_inds = torch.div(max_inds, wx, rounding_mode='floor').float()/ float(hx) # N K 
                v_max_inds = (max_inds % wx).float() / float(wx)
                pos_info = torch.cat( (h_max_inds, v_max_inds), 1) # N 2K 

                pos_info = pos_info.view(nx, 2*self.num_k, 1, 1).float()

                max_inds = max_inds.view(nx, self.num_k, 1, 1).repeat(1, 1, self.tr_num_c, 1) # N K TC 1
            pos_embed = self.pos_embed_convs[li](pos_info) # N (K TC) 1 1

            # make global contexts key / value
            xkey = gkv.clone().view(nx, 1, self.tr_num_c, -1).repeat(1, self.num_k, 1, 1) # N K TC (HW)
            xkey = xkey.gather(3, max_inds) # N K TC 1
            xkey = xkey + pos_embed.view(nx, self.num_k, self.tr_num_c, 1)
            xkey = xkey * max_scr.sigmoid().view(nx, self.num_k, 1, 1) 
            xkey = xkey.view(nx, self.tr_num_c, self.num_k, 1, 1).repeat(1, 1, 1, hx, wx) # N TC K H W
            xkey = torch.cat((xkey, gkv.clone().view(nx, self.tr_num_c, 1, hx, wx)), dim=2) # N TC (K+1) H W

            # vastly simplified implementation of Transformer decoder to avoid heavy computational costs
            # only keeping the core calculation on attention
            att_w = torch.sum(xquery_ * xkey, dim=1, keepdim=True) * float(self.dim_factor) # N 1 (K+1) H W
            att_w = F.softmax(att_w, dim=2)  # N 1 K H W
            xvalue = torch.sum(xkey * att_w, dim=2)  # N TC K H W -> N TC H W
            xvalue = self.inc_norms[li](xvalue)

            # fuse output
            xo = self.dropout_o(F.relu(torch.cat((xquery, xvalue, gkv, gk_mask), dim=1)))

            # refine MFF features
            x = x + self.o_convs[li](xo)
        return x
