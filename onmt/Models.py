import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = onmt.modules.MultiWordEmbedding(opt, dicts)
        
        rnn = lambda: nn.LSTM(input_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=opt.dropout,
                           bidirectional=opt.brnn)
        
        self.rnn = onmt.modules.MultiModule(rnn, len(dicts), share=opt.share_rnn_enc)
    

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = pack(self.word_lut(input[0]), lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs
        
    def switchID(self, srcID):
                
                self.word_lut.switchID(srcID)
                self.rnn.switchID(srcID)
                
    def switchPairID(self, srcID):
                
                return


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts, nPairs=1):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = onmt.modules.MultiWordEmbedding(opt, dicts)
        
        f = lambda: StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
                               
        self.rnn = onmt.modules.MultiModule(f, len(dicts), share=opt.share_rnn_dec) 
        
        attn = lambda : onmt.modules.GlobalAttention(opt.rnn_size)
        self.attn = onmt.modules.MultiModule(attn, nPairs, share=opt.share_attention)
        
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    #~ def load_pretrained_vectors(self, opt):
        #~ if opt.pre_word_vecs_dec is not None:
            #~ pretrained = torch.load(opt.pre_word_vecs_dec)
            #~ self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(0, 1))
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn
        
    
    def switchID(self, tgtID):
                
                self.word_lut.switchID(tgtID)
                self.rnn.switchID(tgtID)
                
    def switchPairID(self, pairID):
                self.attn.switchID(pairID)
                return


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h
            
    # For sampling functions, we need to create an initial input
    # Which is vector of batch_size full of BOS    
    def make_init_input(self, src, volatile=False):
        if isinstance(src, tuple):
                src = src[0]
        batch_size = src.size(1)
        i_size = (1, batch_size)
        
        input_vector = src.data.new(*i_size).fill_(onmt.Constants.BOS)
        return Variable(input_vector, requires_grad=False, volatile=volatile)
    
    # A function to sample from a pre-computed context 
    # We need the context, the initial hidden layer, the initial state (for input feed) and an initial input
    # Options are: using argmax or stochastic, and to save the stochastic actions for reinforcement learning
    def sample_from_context(self, context, init_state, init_hiddens, init_input, 
                                max_length=50, save=False, argmax=True):
                        
        hidden = init_hiddens
        state = init_state
        batch_size = context.size(1) 
        
        # we start from the vector of <BOS>                                
        input_t = init_input
        
        sampled = []
        
        # if we don't save then create volatile variables
        # to save memory
        if not save:
            context = Variable(context.data, volatile=True)
            
            hidden_new = list()
            for h in hidden:
                hidden_new.append(Variable(h.data, volatile=True))
            
            hidden = tuple(hidden_new)
            
            state = Variable(state.data, volatile=True)
            input_t = Variable(input_t.data, volatile=True)
        
        eos_check = init_input[0].data.byte().new(batch_size, 1).zero_()
                        
        pad_mask = init_input[0].data.byte().new(batch_size, 1).zero_()
        
        accumulated_logprob = None
        
        log_probs = []
        
        
        
        for t in xrange(max_length):
            # make a forward pass through the decoder
            state, hidden, attn_t = self.decoder(input_t, hidden, context, state)
            
            state = state.squeeze(0)
            output = self.generator(state) 
            if argmax:
                sample_ = torch.topk(output, 1, dim=1)[1]
                sample = sample_.data
                
                
            else: # Stochastic sampling
                dist = output.exp()
                
                sample_ = dist.multinomial(1)
                
                sample = sample_.data
                
   
            # log_prob of action at time T
            
            log_prob_t = output.gather(1, Variable(sample)).t() # 1 * batch_size
                        
            log_probs.append(log_prob_t) 
            
            # log prob of the samples
            check = (sample == onmt.Constants.EOS)
            
            # update the <eos> check
            eos_check |= check 
            
            # everything after <EOS> is masked as PAD
            sample.masked_fill_(pad_mask, onmt.Constants.PAD)
            
            # update the pad mask
            pad_mask |= check
            
            
            # note: one of the important steps here
            # is to generate the data (tensor) 
            # before making the actual variable
            input_t = Variable(sample.t(), volatile=(not save))
           
            sampled.append(input_t)

            if save:
                assert argmax==False
                            
             # stop sampling when all sentences reach eos 
            if eos_check.sum() == batch_size:
                break
        
        
        # we concatenate them into one single Tensor                                 
        sampled = torch.cat(sampled, 0) # T x B
        
        log_probs = torch.cat(log_probs, 0) # T x B
        
        
        return sampled, log_probs

    # Forward pass :
    # Two (or more) modes: Cross Entropy or Reinforce
    def forward(self, input, mode='xe', max_length=50, gen_greedy=True, timestep_group=8):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
                      
        states = []
        
        outputs = []
        
        hidden = enc_hidden
        state = init_output
        batch_size = tgt.size(1) 
        length = tgt.size(0)
        
        # Cross Entropy training:
        # Using teacher forcing and log-likelihood loss as normally
        if mode == 'xe':
            
            # hiddens : final hidden states for decoder (before linear softmax)
            # dec_hidden: lstm hidden states (c and h for each layer)
            # _attn: attention weights 
            #~ out, dec_hidden, _attn = self.decoder(tgt, enc_hidden,
                                              #~ context, init_output)
            hiddens, dec_hidden, _attn = self.decoder(tgt, enc_hidden,
                  context, init_output)
            
            # hiddens has size T * B * H
            hiddens_split = torch.split(hiddens, timestep_group)
            outputs = []
            
            # we group the computation for faster and more efficient GPU memory
            for i, hidden_group in enumerate(hiddens_split):
                    
                n_steps = hidden_group.size(0)
                hidden_group = hidden_group.view(-1, hidden_group.size(2))
                output_group = self.generator(hidden_group)
                output_group = output_group.view(n_steps, -1, output_group.size(1))
                outputs.append(output_group)
            
            # concatenate into one single tensor
            outputs = torch.cat(outputs, 0)
            
            states = hiddens            
         
            return outputs, states
        elif mode == 'rf':
            
            # initial token (BOS)
            init_input = self.make_init_input(src)
            
            # save=True so that the stochastic actions will be saved for the backward pass
            rl_samples, logprobs = self.sample_from_context(context, init_output, enc_hidden, 
                                            init_input, argmax=False, max_length=min(length + 5, 51), save=True)
            # By default: the baseline is the samples from greedy search
            
            if gen_greedy:
                greedy_samples,  _ = self.sample_from_context(context, init_output, enc_hidden, 
                                                    init_input, argmax=True, max_length=min(length + 5, 51))                                                                                                                                                                 
                return rl_samples, greedy_samples, logprobs
            else:
                return rl_samples, logprobs
        
        else:
            raise NotImplementedError
        
                                              
    def switchLangID(self, srcID, tgtID):
                
        self.encoder.switchID(srcID)
        self.decoder.switchID(tgtID)
        self.generator.switchID(tgtID)
    def switchPairID(self, pairID):
                
        self.decoder.switchPairID(pairID)
        
    # This function needs to look at the dict
    # If the dict at encoder and decoder has the same name -> tie them
    def shareEmbedding(self, dicts):
                
        setIDs = dicts['setIDs']
          
        srcLangs = dicts['srcLangs']
        tgtLangs = dicts['tgtLangs']
        
        tieList = list()

        for (i, srcLang) in enumerate(srcLangs):
            
            for (j, tgtLang) in enumerate(tgtLangs):
                
                if srcLang == tgtLang:
                    
                    tieList.append([i, j])
                    # Tie these embeddings
                    print(' * Tying embedding of encoder and decoder for lang %s' % srcLang)
                    npEnc = self.encoder.word_lut.moduleList[i].weight.nelement()
                    npDec = self.decoder.word_lut.moduleList[j].weight.nelement()
                    assert(npEnc == npDec)
                    self.encoder.word_lut.moduleList[i].weight = self.decoder.word_lut.moduleList[j].weight            
                    
        return tieList
                
                
            

class Generator(nn.Module):
    
    def __init__(self, opt, dicts):
        
        super(Generator, self).__init__()
        
        inputSize = opt.rnn_size
        self.inputSizes = [] 
        self.outputSizes = []
        
        for i in dicts:
            vocabSize = dicts[i].size()
            self.outputSizes.append(vocabSize)
            self.inputSizes.append(inputSize)
            
        self.linear = onmt.modules.MultiLinear(self.inputSizes, self.outputSizes)
        self.lsm = nn.LogSoftmax()
                            
    def forward(self, input):
        
        output = self.lsm(self.linear(input))
        return output
        
    
    def switchID(self, tgtID):
        
        self.linear.switchID(tgtID)


def NMTCriterion(dicts, cuda=True):
    
    crits = dict()
    for i in dicts:
        vocabSize = dicts[i].size()
        
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        if cuda:
            crit.cuda()
        
        crits[i] = crit
    
    return crits
