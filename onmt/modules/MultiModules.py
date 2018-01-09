import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import copy

class MultiWordEmbedding(nn.Module):
    
    def __init__(self, opt, dicts):
        
        super(MultiWordEmbedding, self).__init__()
        self.inputSize = opt.word_vec_size
        self.moduleList = nn.ModuleList()
        
        for i in dicts:
            vocabSize = dicts[i].size()
            
            embedding = nn.Embedding(vocabSize,
                                     self.inputSize,
                                     padding_idx=onmt.Constants.PAD)
            self.moduleList.append(embedding)
        
        
        self.currentID = 0
        
    
    def switchID(self, idx):
        
        assert idx >= 0 and idx < len(self.moduleList)
        self.currentID = idx
        
    
    def forward(self, input):
        
        lookup = self.moduleList[self.currentID]
        return lookup(input)


class MultiLinear(nn.Module):
    
    def __init__(self, inputSizes, outputSizes, bias=True):
        
        super(MultiLinear, self).__init__()
        
        self.inputSizes = inputSizes
        self.outputSizes = outputSizes
        
        assert len(self.inputSizes) == len(self.outputSizes)
        
        self.moduleList = nn.ModuleList()
        
        
        for i in xrange(len(self.inputSizes)):
            
            linear = nn.Linear(inputSizes[i], outputSizes[i], bias=bias)
            self.moduleList.append(linear)
            
        self.currentID = 0
        
    
    def switchID(self, idx):
    
        assert idx >= 0 and idx < len(self.moduleList)
        self.currentID = idx
        
    def forward(self, input):
    
        linear = self.moduleList[self.currentID]
        return linear(input)




class MultiCloneModule(nn.Module):
    
    def __init__(self, m, dicts, share=False):
        
        super(MultiCloneModule, self).__init__()
        
        self.moduleList = nn.ModuleList()
        self.moduleList.append(m)
        
        for i in range(1, len(dicts)):
            clone = copy.deepCopy(m)
            
            if share:
                clone.weight = m.weight
                
            self.moduleList.append(clone)
            
        self.currentID = 0
        
    def switchID(self, idx):
        
        assert idx >= 0 and idx < len(self.moduleList)
        self.currentID = idx
        
    def forward(self, input):
        
        module = self.moduleList[self.currentID]
        return module(input)
            
class MultiModule(nn.Module):
    
    def __init__(self, m, nModules, share=False):
    
        super(MultiModule, self).__init__()
        self.moduleList = nn.ModuleList()
        self.share = share
        
        for i in range(nModules):
            
            if not self.share or i == 0:
                module = m()
                self.moduleList.append(module)
                
        
        self.currentID = 0
        
        
    def switchID(self, idx):
        
        if not self.share:
            assert idx >= 0 and idx < len(self.moduleList)
            self.currentID = idx
        
    def forward(self, *input):
        
        module = self.moduleList[self.currentID]
        return module(*input)
        
    def current(self):
        return self.moduleList[self.currentID]
