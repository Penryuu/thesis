import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

class BiC(nn.Module):
    def __init__(self, lr, scheduling, lr_decay_factor, weight_decay, batch_size, epochs):
        super(BiC, self).__init__()
        self.beta = torch.nn.Parameter(torch.ones(1)).cuda()
        self.gamma = torch.nn.Parameter(torch.zeros(1)).cuda()
        self.lr = lr
        self.scheduling = scheduling
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.class_specific = False
        self.batch_size = batch_size
        self.epochs = epochs
        self.bic_flag = False