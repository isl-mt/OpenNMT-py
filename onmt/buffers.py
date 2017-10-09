from __future__ import division

import math
import torch
from torch.autograd import Variable

import onmt

class GradientBuffer(object):
    
    def __init__(self, model):
        self.buffers = dict()
        
        self.model = model
        
        self.init_buffer()

    def add_buffer(self):
        
        i = 0
        for p in self.model.parameters():
            i = i + 1
            if p.grad is not None:
                if self.buffers[i] is not None:
                    self.buffers[i].add_(p.grad.data)
                else:
                    self.buffers[i] = p.grad.data.clone()
                
    def init_buffer(self):
        i = 0
        for p in self.model.parameters():
            i = i + 1
            self.buffers[i] = None
                
    def accumulate_buffer(self, scale=1):
        
        i = 0
        for p in self.model.parameters():
            i = i + 1
            if p.grad is not None:
                if self.buffers[i] is not None:
                    p.grad.data.add_(self.buffers[i].div_(scale))
            elif self.buffers[i] is not None:
                data = p.grad.data
                data.div_(scale)
                p.grad = Variable(data)
            
        
    def zero_buffer(self):
        for i in self.buffers:
            if self.buffers[i] is not None:
                self.buffers[i].zero_()
