#~ """
#~ Global attention takes a matrix and a query vector. It
#~ then computes a parameterized convex combination of the matrix
#~ based on the input query.
#~ 
#~ 
        #~ H_1 H_2 H_3 ... H_n
          #~ q   q   q       q
            #~ |  |   |       |
              #~ \ |   |      /
                      #~ .....
                  #~ \   |  /
                          #~ a
#~ 
#~ Constructs a unit mapping.
    #~ $$(H_1 + H_n, q) => (a)$$
    #~ Where H is of `batch x n x dim` and q is of `batch x dim`.
#~ 
    #~ The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:
#~ 
#~ """
#~ 
import torch
import torch.nn as nn
#~ 
#~ 
class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.linear_to_one = nn.Linear(dim, 1, bias=True)
        self.tanh = nn.Tanh()
        self.mlp_tanh = nn.Tanh()
        self.mask = None
        
        # For context gate
        self.linear_cg = nn.Linear(dim*2, dim, bias=True)
        self.sigmoid_cg = nn.Sigmoid()

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        bsize = context.size(0)
        seq_length = context.size(1)
        dim = context.size(2)
        
        # project the hidden state (query)
        targetT = self.linear_in(input).unsqueeze(1)  # batch x 1 x dim
        
        # project the context (keys and values)
        reshaped_ctx = context.contiguous().view(bsize * seq_length, dim)
        
        projected_ctx = self.linear_context(reshaped_ctx)
        
        projected_ctx = projected_ctx.view(bsize, seq_length, dim)
        
        # MLP attention model
        
        repeat = targetT.expand_as(projected_ctx)
        sum_query_ctx = repeat + projected_ctx 
        sum_query_ctx = sum_query_ctx.view(bsize * seq_length, dim)
        
        mlp_input = self.mlp_tanh(sum_query_ctx)
        mlp_output = self.linear_to_one(mlp_input)
        
        mlp_output = mlp_output.view(bsize, seq_length, 1)
        attn = mlp_output.squeeze(2)
        #~ attn = mlp_output.squeeze(1).view(bsize, seq_length)
        #~ attn = mlp_output.view(bsize, seq_length) # batch x sourceL

        # Get attention
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
            self.mask = None
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)
        
        #ContextGate
        contextGate = self.sigmoid_cg(self.linear_cg(contextCombined))
        inputGate = 1 - contextGate
        
        gatedContext = weightedContext * contextGate
        gatedInput = input * inputGate
        gatedContextCombined = torch.cat((gatedContext, gatedInput), 1)
        

        contextOutput = self.tanh(self.linear_out(gatedContextCombined))

        return contextOutput, attn


#~ import torch
#~ import torch.nn as nn
#~ 
#~ 
#~ class GlobalAttention(nn.Module):
    #~ def __init__(self, dim):
        #~ super(GlobalAttention, self).__init__()
        #~ self.linear_in = nn.Linear(dim, dim, bias=False)
        #~ self.sm = nn.Softmax()
        #~ self.linear_out = nn.Linear(dim*2, dim, bias=False)
        #~ self.tanh = nn.Tanh()
        #~ self.mask = None
#~ 
    #~ def applyMask(self, mask):
        #~ self.mask = mask
#~ 
    #~ def forward(self, input, context):
        #~ """
        #~ input: batch x dim
        #~ context: batch x sourceL x dim
        #~ """
        #~ targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
#~ 
        #~ # Get attention
        #~ attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        #~ if self.mask is not None:
            #~ attn.data.masked_fill_(self.mask, -float('inf'))
        #~ attn = self.sm(attn)
        #~ attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
#~ 
        #~ weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        #~ contextCombined = torch.cat((weightedContext, input), 1)
#~ 
        #~ contextOutput = self.tanh(self.linear_out(contextCombined))
#~ 
        #~ return contextOutput, attn
