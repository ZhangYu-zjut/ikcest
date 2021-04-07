import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
import numpy as np
class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.ratio = args.ratio
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.GRU1 = nn.GRU(self.m, self.hidR)
        self.residual_window = args.residual_window
        
        self.mask_mat =Parameter(torch.Tensor(np.random.randn(self.m, self.m)*np.sqrt(2.0/self.m)))
        
        
        #print("mask_mat",self.mask_mat)
        #print("mask_mat,shape",self.mask_mat.shape)

        self.adj = data.adj
        self.test = torch.Tensor(self.m,self.m)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1);
        self.output = None
        
        ##
        if (args.output_fun == 'sigmoid'):
            #self.output = F.sigmoid
            self.output = torch.sigmoid
        ##
        if (args.output_fun == 'relu'):
            #self.output = F.sigmoid
            self.output = torch.relu
        if (args.output_fun == 'tanh'):
            #self.output = F.tanh
            self.output = torch.tanh

    def forward(self, x):
        # x: batch x window (self.P) x #signal (m)
        # first transform
        masked_adj = self.adj * self.mask_mat
    
        x = x.matmul(masked_adj)
        
    
        # RNN
        # r: window (self.P) x batch x #signal
        r = x.permute(1, 0, 2).contiguous()
        _, r = self.GRU1(r)
        
        r = self.dropout(torch.squeeze(r, 0))
    
        res = self.linear1(r)

        #residual
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window);
            z = self.residual(z);
            z = z.view(-1,self.m);
            res = res * self.ratio + z;

        if self.output is not None:
            res = self.output(res).float()
        
        return res
