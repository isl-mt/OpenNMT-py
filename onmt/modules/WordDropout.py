#~ """
#~ Word Dropout involves applying a mask over the discrete token sequence (words)
#  Same word types are masked similarly all over the sequence (word tying)

import torch
import torch.nn as nn
import onmt
from torch.autograd import Variable

class WordDropout(nn.Module):
    
    def __init__(self, p=0.0):
        
        super(WordDropout, self).__init__()
        
        # dropout probability
        self.p = p 
        print(self.p)
    
    
        
    def forward(self, input):
        
        if self.training and self.p > 0:
            # allocate a new mask
            i_size = input.data.size()
            mask = input.data.new(*i_size).float()
            mask.fill_(1 - self.p)
            
            # inplace bernoulli generation for dropout
            mask.bernoulli_().div(1 - self.p)
            
            #~ print(mask)
            
            # tying probability over words: (cheating)
            
            #~ for b in xrange(input.size(1)): # batch_size
                #~ for t in xrange(input.size(0)): # sequence length
                    #~ x = input.data[t][b]
                    #~ for t_next in xrange(t+1, input.size(0)):
                        #~ if input.data[t_next][b] == x:
                            #~ mask[t_next][b] = mask[t][b]
            output = input * mask
            
            return output
        
        else:
            return input
    
