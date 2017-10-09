import torch, sys
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import random
import rnnlib 

#~ class CustomEncoder(nn.Module):
#~ 
    #~ def __init__(self, opt, dicts, embeddings):
        #~ self.layers = opt.layers
        #~ self.num_directions = 2 if opt.brnn else 1
        #~ assert opt.rnn_size % self.num_directions == 0
        #~ self.hidden_size = opt.rnn_size // self.num_directions
        #~ input_size = opt.word_vec_size
        #~ dropout_value = opt.dropout 
#~ 
        #~ super(Encoder, self).__init__()
        #~ 
        #~ self.word_lut = embeddings
        #~ self.rnn = nn.LSTM(input_size, self.hidden_size,
                           #~ num_layers=opt.layers,
                           #~ dropout=dropout_value,
                           #~ bidirectional=opt.brnn)
                           #~ 
    #~ def load_pretrained_vectors(self, opt):
        #~ if opt.pre_word_vecs_enc is not None:
            #~ pretrained = torch.load(opt.pre_word_vecs_enc)
            #~ self.word_lut.weight.data.copy_(pretrained)
#~ 
    #~ def forward(self, input, hidden=None):
        #~ if isinstance(input, tuple):
            #~ # Lengths data is wrapped inside a Variable.
            #~ lengths = input[1].data.view(-1).tolist()
            #~ emb = pack(self.word_lut(input[0]), lengths)
        #~ else:
            #~ emb = self.word_lut(input)
        #~ outputs, hidden_t = self.rnn(emb, hidden)
        #~ if isinstance(input, tuple):
            #~ outputs = unpack(outputs)[0]
        #~ return hidden_t, outputs
        
class Encoder(nn.Module):

    def __init__(self, opt, dicts, embeddings, custom=False):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size
        dropout_value = opt.dropout 

        super(Encoder, self).__init__()
        
        self.word_lut = embeddings
        self.custom = custom
        
        if custom != True:
            self.rnn = nn.LSTM(input_size, self.hidden_size,
                               num_layers=opt.layers,
                               dropout=dropout_value,
                               bidirectional=opt.brnn)
        else:
            self.rnn = rnnlib.RecurrentLayer(rnnlib.Cells.LSTMCell, input_size, opt.rnn_size,
                                             num_layers=opt.layers, dropout=dropout_value,
                                             bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)
    
    def init_hidden(self, emb, batch_size):
        
        hidden = []
        
        for i in xrange(self.layers):
            
            for j in xrange(self.num_directions):
                # allocate the initial hidden state
                h = Variable(emb.data.new(batch_size, self.hidden_size).zero_())
                c = Variable(emb.data.new(batch_size, self.hidden_size).zero_())
                hidden.append((h,c))
        
        
        return hidden
        

    def forward(self, input, hidden=None):
        #~ print isinstance(input, tuple)
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            batch_size = input[0].size(1)
            emb = pack(self.word_lut(input[0]), lengths)
            emb_data = emb[0]
        else:
            batch_size = input.size(1)
            emb = self.word_lut(input)
            emb_data = emb
        
        if hidden is None and self.custom:
            hidden = self.init_hidden(emb_data, batch_size)
        
        outputs, hidden_t = self.rnn(emb, hidden)
        #~ print(outputs)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout, custom=False):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        cell = nn.LSTMCell
        if custom == True:
            cell = rnnlib.Cells.LSTMCell

        for i in range(num_layers):
            self.layers.append(cell(input_size, rnn_size))
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

    def __init__(self, opt, dicts, embeddings, custom=False):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        #~ self.word_lut = nn.Embedding(dicts.size(),
                                     #~ opt.word_vec_size,
                                     #~ padding_idx=onmt.Constants.PAD)
        self.word_lut = embeddings
        self.rnn = StackedLSTM(opt.layers, input_size,
                               opt.rnn_size, opt.dropout, custom=custom)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size
        self.input_size = input_size
        
    def free_mask(self):
        self.attn.free_mask()

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

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
        
class RecurrentEncoderDecoder(nn.Module):
    
    def __init__(self, encoder, decoder, generator):
        super(RecurrentEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
    # For sampling functions, we need to create an initial input
    # Which is vector of batch_size full of BOS    
    def make_init_input(self, src, volatile=False):
        if isinstance(src, tuple):
                src = src[0]
        batch_size = src.size(1)
        i_size = (1, batch_size)
        
        input_vector = src.data.new(*i_size).fill_(onmt.Constants.BOS)
        return Variable(input_vector, requires_grad=False, volatile=volatile)
        
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
    

class NMTModel(RecurrentEncoderDecoder):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__(encoder, decoder, generator)
        
        # For reinforcement learning 
        self.saved_actions = []
        self.rewards = [] # remember that this is cumulative (sum of rewards from start to end)
    
    def tie_weights(self):
        self.decoder.word_lut.weight = self.generator.net[0].weight
        
    def tie_join_embeddings(self):
        self.encoder.word_lut.weight = self.decoder.word_lut.weight
    
    # Samping function (mostly for debugging)         
    def sample(self, input, max_length=50, argmax=True):
                        
                                
        src = input
        # we don't care about tgt here
        
        enc_hidden, context = self.encoder(src)
        
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                self._fix_enc_hidden(enc_hidden[1]))
        
        sampled = []
        
        # we start from the vector of <BOS>                                
        init_input = self.make_init_input(src, volatile=True)
        
        # For input feeding initial output
        init_output = self.make_init_decoder_output(context)

        sampled,  logprob = self.sample_from_context(context, init_output, enc_hidden, 
                                                  init_input, argmax=argmax, max_length=max_length)
        
        return sampled
                
                
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
        
        if not save:
            context = Variable(context.data, volatile=True)
            
            hidden_new = list()
            for h in hidden:
                hidden_new.append(Variable(h.data, volatile=True))
            
            hidden = tuple(hidden_new)
            
            state = Variable(state.data, volatile=True)
            input_t = Variable(input_t.data, volatile=True)
        
        tensor_check = init_input[0].data.byte().new(batch_size, 1).zero_()
                        
        last_mask = init_input[0].data.byte().new(batch_size, 1).zero_()
        
        accumulated_logprob = None
        
        
        for t in xrange(max_length):
            # make a forward pass through the decoder
            state, hidden, attn_t = self.decoder(input_t, hidden, context, state)
            
            state = state.squeeze(0)
            output = self.generator(state) 
            if argmax:
                sample = torch.topk(output, 1, dim=1)[1].data
                
                
            else: # Stochastic sampling
                dist = output.exp()
                
                #~ sample = torch.multinomial(dist, 1) # batch_size x 1
                sample_ = dist.multinomial(1)
                
                sample = sample_.data
            
            # log prob of the samples
            #~ logprob = output.index_select(1, input_t.squeeze(0))        # batch_size * 1 
                    
            if save:
                assert argmax==False
                self.saved_actions.append(sample_)
                
            check = (sample == onmt.Constants.EOS)

            tensor_check |= check 
            
            sample.masked_fill_(last_mask, onmt.Constants.PAD)
            
            last_mask |= check
          
            break_signal = False
            
            if tensor_check.sum() == batch_size:
                break_signal = True
            
            
            input_t = Variable(sample.t())
            
            
            #~ logprob.masked_fill(tensor_check.squeeze(0), 0)
            
            #~ if accumulated_logprob is None:
                    #~ accumulated_logprob = logprob
            #~ else:
                    #~ accumulated_logprob += logprob
            
            sampled.append(input_t)
            # if all sentences are finished (reach EOS)
            if break_signal:
                break
        
        # gather the lengths of the samples                                                
        #~ lengths = [len(sampled) for i in xrange(batch_size)]
 
        #~ for i in xrange(batch_size):
                        
            #~ for t in xrange(len(sampled)):
                    #~ if sampled[t].data[0][i] == onmt.Constants.EOS:
                            #~ lengths[i] = t + 1
                            #~ break
        
        # we concatenate them into one single Tensor                                 
        sampled = torch.cat(sampled, 0)
        
        #~ print(lengths)
        
        #~ return sampled, lengths, accumulated_logprob
        return sampled, accumulated_logprob
                
                
    # Forward pass :
    # Two (or more) modes: Cross Entropy or Reinforce
    def forward(self, input, mode='xe', max_length=50, gen_greedy=True, timestep_group=8):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        # Exclude <s> from targets for labels
        tgt_label = input[1][1:]
        
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
                        
                
                
            return outputs, states

        # REINFORCE as in
        # Self critical Reinforcement Learning
        elif mode == 'rf' or mode == 'reinforce':
            # initial token (BOS)
            init_input = self.make_init_input(src)
            
            # save=True so that the stochastic actions will be saved for the backward pass
            rl_samples, logprobs = self.sample_from_context(context, init_output, enc_hidden, 
                                            init_input, argmax=False, max_length=min(length + 5, 51), save=True)
            # By default: the baseline is the samples from greedy search
            
            if gen_greedy:
                greedy_samples,  _ = self.sample_from_context(context, init_output, enc_hidden, 
                                                    init_input, argmax=True, max_length=min(length + 5, 51))                                                                                                                                                                 
                return rl_samples, greedy_samples
            else:
                return rl_samples
        
        # Actor Critic
        # We only sample, the baseline scores are handled by another critic model
        elif mode == 'ac':
            
            # initial token (BOS)
            init_input = self.make_init_input(src)
        
            # save=True so that the stochastic actions will be saved for the backward pass
            rl_samples, logprobs = self.sample_from_context(context, init_output, enc_hidden, 
                                            init_input, argmax=False, max_length=min(length + 5, 50), save=True)

            return rl_samples
            
            
class Generator(nn.Module):
    def __init__(self, inputSize, dicts):
            
            super(Generator, self).__init__()
            
            self.inputSize = inputSize
            self.outputSize = dicts.size()
            
            self.net = nn.Sequential(
            nn.Linear(inputSize, self.outputSize),
            nn.LogSoftmax())

    def forward(self, input):
            return self.net(input)


# The critic generator is only a linear regressor         
class CriticGenerator(nn.Module):
    
    def __init__(self, inputSize):
            
        super(CriticGenerator, self).__init__()
        
        self.inputSize = inputSize
        
        # map the hidden state to a value
        self.net = nn.Linear(inputSize, 1)
            

    def forward(self, input):
        return self.net(input)


class CriticModel(RecurrentEncoderDecoder):

    def __init__(self, encoder, decoder, generator):
        super(CriticModel, self).__init__(encoder, decoder, generator)
        
    def forward(self, src, samples ):
        
        # forward through the encoder first
        enc_hidden, context = self.encoder(src)
        
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
                      
        states = []
        
        outputs = []
        
        hidden = enc_hidden
        state = init_output
        batch_size = samples.size(1) 
        length = samples.size(0)
        
        # forward through the critic decoder 
        hiddens, dec_hidden, _attn = self.decoder(samples, enc_hidden,
                  context, init_output)
        
        # reshae to batch * length x hidden_size
        reshaped_hiddens = hiddens.view(-1, hiddens.size(-1)) 
        
        # batch * length x 1
        critic_output = self.generator(reshaped_hiddens) 
        
        # reshape back to length x batch x 1
        critic_output = critic_output.view(length, batch_size) 
        
        return critic_output
