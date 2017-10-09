from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
from onmt.metrics.gleu import sentence_gleu
from onmt.metrics.sbleu import sentence_bleu
from onmt.metrics.bleu import moses_multi_bleu
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import random 
import numpy as np

def addone(f):
    for line in f:
        yield line
    yield None

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-eval_batch_size', type=int, default=8,
                    help='Maximum batch size for decoding eval')
parser.add_argument('-tie_weights', action='store_true',
                    help='Tie the weights of the decoder embedding and logistic regression layer')
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-reinforce_rate', type=float, default=0.0,
                    help='Rate of using reinforcement learning during training')
parser.add_argument('-reinforce_sampling_number', type=int, default=1,
                    help='Number of samples during reinforcement learning')
parser.add_argument('-reinforce_metrics', default='gleu',
                    help='Metrics for reinforcement learning. Default = gleu')
# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=1,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=1000,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-reset_optim', action='store_true',
                    help='Use a bidirectional encoder')
# pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-seed', default=9999, nargs='+', type=int,
                    help="Seed for deterministic runs.")

parser.add_argument('-log_interval', type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument('-save_every', type=int, default=-1,
                    help="Save every this interval.")
parser.add_argument('-sample_every', type=int, default=2000,
                    help="Save every this interval.")

parser.add_argument('-valid_src', default='',
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', default='',
                    help="Path to the validation target data")                   
opt = parser.parse_args()

print(opt)

if opt.reinforce_metrics == 'gleu':
    score = sentence_gleu
elif opt.reinforce_metrics == 'sbleu':
    score = sentence_bleu


if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

torch.manual_seed(opt.seed)

gradient_buffers = dict()

def add_buffer(model):
    
    i = 0
    for p in model.parameters():
        i = i + 1
        if p.grad is not None:
            if gradient_buffers[i] is not None:
                gradient_buffers[i].add_(p.grad.data)
            else:
                gradient_buffers[i] = p.grad.data.clone()
            
def init_buffer(model):
    i = 0
    for p in model.parameters():
        i = i + 1
        gradient_buffers[i] = None
            
def accumulate_buffer(model, n):
    
    i = 0
    for p in model.parameters():
        i = i + 1
        #~ if gradient_buffers[i] 
        if p.grad is not None:
            if gradient_buffers[i] is not None:
                p.grad.data.add_(gradient_buffers[i].div_(n))
        elif gradient_buffers[i] is not None:
            data = p.grad.data
            data.div_(n)
            p.grad = Variable(data)
        
    #~ pass
    
def zero_buffer():
    for i in gradient_buffers:
        if gradient_buffers[i] is not None:
            gradient_buffers[i].zero_()

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def eval_translate(model, dicts, srcFile, tgtFile, beam_size=1, bpe=True):
        
        print(" * Translating file %s " % srcFile )
        if len(srcFile) == 0:
            return 0
        # initialize the translator for beam search
        translator = onmt.InPlaceTranslator(model, dicts, beam_size=beam_size, 
                                                                                batch_size=opt.eval_batch_size, 
                                                                                cuda=len(opt.gpus) >= 1)
        
        srcBatch = []
        
        count = 0
        
        # we print translations into temp files
        outF = tempfile.NamedTemporaryFile()
        outRef = tempfile.NamedTemporaryFile()
        
        nLines = len(open(srcFile).readlines())
        
        inFile = open(srcFile)
        

        for line in addone(inFile):
            if line is not None:
                srcTokens = line.split()
                srcBatch += [srcTokens]
                if len(srcBatch) < opt.eval_batch_size:
                    continue
            
            if len(srcBatch) == 0:
                break        
                
            predBatch, predScore, goldScore = translator.translate(srcBatch)
            
            for b in range(len(predBatch)):
                count += 1
                decodedSent = " ".join(predBatch[b][0])
                
                if bpe:
                    decodedSent = decodedSent.replace('@@ ', '')
                
                outF.write(decodedSent + "\n")
                outF.flush()
                
                sys.stdout.write("\r* %i/%i Sentences" % (count , nLines))
                sys.stdout.flush()
            
            srcBatch = []
            
        print("\nDone")
        refFile = open(tgtFile)
        
        for line in addone(refFile):
            if line is not None:
                line = line.strip()
                if bpe:
                    line = line.replace('@@ ', '')
                outRef.write(line + "\n")
                outRef.flush()
        
        # compute bleu using external script
        bleu = moses_multi_bleu(outF.name, outRef.name)
        refFile.close()
        inFile.close()
        outF.close()
        outRef.close()
        # after decoding, switch model back to training mode
        model.train()
        
        return bleu

def compute_score(samples, lengths, ref, dicts, batch_size, average=True):
        
        # probably faster than gpu ?
        #~ samples = samples.cpu()
        
        sdata = samples.data.cpu()
        rdata = ref.data.cpu()
        
        tgtDict = dicts['tgt']
        
        s = torch.Tensor(batch_size)
        
        for i in xrange(batch_size):
            
            sampledIDs = sdata[:,i]
            refIDs = rdata[:,i]
            
            sampledWords = tgtDict.convertTensorToLabels(sampledIDs, onmt.Constants.EOS)
            refWords = tgtDict.convertTensorToLabels(refIDs, onmt.Constants.EOS)
            
            s[i] = score(refWords, sampledWords)
            assert(len(sampledWords) == lengths[i])
            
        s = s.cuda()
            
        return s
 
def sample(model, batch, dicts, eval=False):
        
        # output of sampling function is a list 
        # containing sampled indices for each time step
        # ( padded to the right )
        #~ model.eval()
        if eval:
            model.eval()
        print("\nSampling ... ")
        
        src = batch[0]
        
        # I wrap a new variable so that the sampling process
        # doesn't record any history (more memory efficient ?)
        variable = Variable(src[0].data, volatile=True)
        length = Variable(src[1].data, volatile=True)
        
        sampled_sequence, lengths = model.sample((variable, length), argmax=False)
        
        tgtDict = dicts['tgt']
        srcDict = dicts['src']
        
        batch_size = sampled_sequence.size(1)
        
        indices = random.sample(range(batch_size), min(10, batch_size))
        
        ref = batch[1][1:]
        
        for idx in indices:
                
            tgtIds = sampled_sequence.data[:,idx]
            
            tgtWords = tgtDict.convertTensorToLabels(tgtIds, onmt.Constants.EOS)
            
            sampledSent = " ".join(tgtWords)
            
            print "SAMPLE :", sampledSent
            
            refIds = ref.data[:,idx]
            
            refWords = tgtDict.convertTensorToLabels(refIds, onmt.Constants.EOS)
            
            refSent = " ".join(refWords)
            
            print "   REF :", refSent
            
            s = score(refWords, tgtWords)
            
            print "Score =", s
        
        if eval:
            model.train()
            
        print("\n")
    

def eval(model, data):
    total_loss = 0
    total_words = 0
    

    model.eval()
    for i in range(len(data)):
        # exclude original indices
        batch = data[i][:-1]
        loss, _, _ = model(batch, eval=True)
        # exclude <s> from targets
        targets = batch[1][1:]
        total_loss += loss
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words


def duplicateBatch(batch, volatile=False):
    
        
        src = Variable(batch[0][0].data, volatile=volatile)
        tgt = Variable(batch[1].data, volatile=volatile)
        length = Variable(batch[0][1].data, volatile=volatile)
        #~ indices = batch[2]
        
        return (src, length), tgt


def trainModel(model, trainData, validData, dataset, optim):
    
    model.train()
    
    dicts = dataset['dicts']

    start_time = time.time()

    def trainEpoch(epoch, batchOrder=None):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        
        if not batchOrder:
                    batchOrder = torch.randperm(len(trainData))

        total_loss_xe, total_words_xe = 0, 0
        report_loss_xe, report_tgt_words_xe = 0, 0
        report_src_words, report_tgt_words = 0, 0
        start = time.time()
        nSamples = len(trainData)
        
        i = 0 # This is the sample index
        counter = 0 # This is the counter for each sample
        last_batch = None
        greedy_reward = 0
        
        train_mode = 'xe'
        
        #~ for i in range(nSamples):
        while i < nSamples:

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            #~ if counter == 0 :
            batch = trainData[batchIdx][:-1]
            #~ else:
                            #~ batch = last_batch
            #~ last_batch = batch
            batch_size = batch[1].size(1)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            num_words = targets.data.ne(onmt.Constants.PAD).sum()

            model.zero_grad()
            
            if counter == 0:
                            # important: only zero grad model at counter = 0
                            zero_buffer()
                            train_mode = 'xe'
                            if random.random() < opt.reinforce_rate:
                                train_mode = 'rf'
                        
                        # For Cross Entropy mode training
            if train_mode == 'xe':
                            # Loss is computed by nll already
                            loss, _, _ = model(batch, mode=train_mode)
                            loss_value = loss.data[0]
                            loss.div(batch_size).backward()
            
                            loss_xe = loss_value
                            
                            report_loss_xe += loss_xe
                            report_tgt_words_xe += num_words
                            
                            total_loss_xe += loss_xe
                            total_words_xe += num_words
                            optim.step()
                            
                            counter = 100000
                            
                        # For reinforcement learning mode
            elif train_mode == 'rf':
                ref = batch[1][1:]
                # Monte-Carlo actions and greedy actions to be sampled
                
                #~ greedy_actions, greedy_lengths = None, None
                
                
                # For the first sample, we also generate the greedy one
                if counter == 0:
                    rl_actions, rl_lengths, greedy_actions, greedy_lengths = model(batch, mode=train_mode)
                    # Reward for samples from greedy search
                    greedy_reward = compute_score(greedy_actions, greedy_lengths, ref, dicts, batch_size) 
                else:
                    #~ new_batch = duplicateBatch(batch)
                    rl_actions, rl_lengths = model(batch, mode=train_mode, gen_greedy=False)
                
                # reward for samples from stochastic function
                sampled_reward = compute_score(rl_actions, rl_lengths, ref, dicts, batch_size) 
                    
                # the REINFORCE reward to be the difference between MC and greedy
                # should we manually divide rewards by batch size ? since we may have different batch size
                rf_rewards = (sampled_reward - greedy_reward) / batch_size 
                
                
                # centralize the rewards to make learning faster ?
                rf_rewards = (rf_rewards - rf_rewards.mean()) / (rf_rewards.std() + np.finfo(np.float32).eps)
                
                # Reward cumulative backward:
                length = rl_actions.size(0)
                for t in xrange(length):
                    
                    reward_t = rf_rewards.clone()
                    
                    # a little hack here: since we only have reward at the last step
                    # so the cumulative is the reward itself at every time step
                    for b in xrange(batch_size):
                        # important: reward for PAD tokens = 0
                        if t+1 > rl_lengths[b]:
                            reward_t[b] = 0
                            
                    model.saved_actions[t].reinforce(reward_t.unsqueeze(1))
                
                # We backward from stochastic actions, ignoring the gradOutputs
                torch.autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
                # Free the action list.    
                model.saved_actions = []
                #~ optim.step()
                add_buffer(model)
                
                # update only after the loop
                counter = counter + 1
                
                if counter == opt.reinforce_sampling_number:
                    model.zero_grad()
                    accumulate_buffer(model, opt.reinforce_sampling_number)
                    optim.step()    
                
                
                del model.saved_actions[:]
                
                del rl_actions
                #~ if n == 0:
                    #~ del greedy_actions
                #~ del rl_lengths
                del rf_rewards
                            
            repeat_flag = False
            if counter >= opt.reinforce_sampling_number:
                            i = i + 1 # increase the indexing counter                     
                            counter = 0
            else:
                            repeat_flag = True
                            
            if not repeat_flag:
                index = i - 1
                report_src_words += batch[0][1].data.sum()
                report_tgt_words += num_words

                # Sample at the end of epoch (experimental):
                if index % opt.sample_every == -1 % opt.sample_every:
                    sample(model, batch, dicts)

                
                if index == 0 or (index % opt.log_interval == -1 % opt.log_interval):
                        print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f; lr: %1.6f; " +
                                     "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                                    (epoch, index+1, len(trainData),
                                     math.exp(report_loss_xe / (report_tgt_words_xe + 1e-6)),
                                     optim.lr,
                                     report_src_words/(time.time()-start),
                                     report_tgt_words/(time.time()-start),
                                     time.time()-start_time))

                        report_loss_xe, report_tgt_words_xe = 0, 0
                        report_src_words = 0
                        report_tgt_words = 0
                        start = time.time()
                
                if opt.save_every > 0 and index % opt.save_every == -1 % opt.save_every :
                    valid_loss = eval(model, validData).data[0]
                    valid_ppl = math.exp(min(valid_loss, 100))
                    valid_bleu = eval_translate(model, dicts, opt.valid_src, opt.valid_tgt)
                    print('Validation perplexity: %g' % valid_ppl)
                    print('Validation BLEU: %.2f' % valid_bleu)
                    
                    model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                                                else model.state_dict())
                    model_state_dict = {k: v for k, v in model_state_dict.items()}
                    
                    #  drop a checkpoint
                    ep = float(epoch) - 1 + (index + 1) / nSamples
                    checkpoint = {
                            'model': model_state_dict,
                            #~ 'generator': generator_state_dict,
                            'dicts': dataset['dicts'],
                            'opt': opt,
                            'epoch': ep,
                            'iteration' : index,
                            'batchOrder' : batchOrder,
                            'optim': optim
                    }
                    
                    file_name = '%s_ppl_%.2f_bleu_%.2f_e%.2f.pt'
                    print('Writing to ' + file_name % (opt.save_model, valid_ppl, valid_bleu, ep))
                    torch.save(checkpoint,
                                         file_name
                                         % (opt.save_model, valid_ppl, valid_bleu, ep))
            #~ if counter >= opt.reinforce_sampling_number:
                            #~ i = i + 1 # increase the indexing counter                     
                            #~ counter = 0
        return total_loss_xe / (total_words_xe + 1e-6)

    for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
        print('')

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)

        #  (2) evaluate on the validation set
        valid_loss = eval(model, validData).data[0]
        valid_ppl = math.exp(min(valid_loss, 100))
        valid_bleu = eval_translate(model, dicts, opt.valid_src, opt.valid_tgt)
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation BLEU: %.2f' % valid_bleu)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_ppl, epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()}
        #~ generator_state_dict = (model.generator.module.state_dict()
                                #~ if len(opt.gpus) > 1
                                #~ else model.generator.state_dict())
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            #~ 'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'iteration' : -1,
            'batchOrder' : None,
            'optim': optim
        }
        
                
        file_name = '%s_ppl_%.2f_bleu_%.2f_e%.2f.pt'
        print('Writing to ' + file_name % (opt.save_model, valid_ppl, valid_bleu, epoch))
        torch.save(checkpoint,
                   file_name % (opt.save_model, valid_ppl, valid_bleu, epoch))


def main():
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    print("Done")
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.encoder_type == "text":
        encoder = onmt.Models.Encoder(opt, dicts['src'])
    elif opt.encoder_type == "img":
        encoder = onmt.modules.ImageEncoder(opt)
        assert("type" not in dataset or dataset["type"] == "img")
    else:
        print("Unsupported encoder type %s" % (opt.encoder_type))

    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    generator = onmt.Models.Generator(opt.rnn_size, dicts['tgt'])
    
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    model = onmt.Models.NMTModel(encoder, decoder, generator)
    

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        #~ chk_model = checkpoint['model']
        #~ generator.load_state_dict(checkpoint['generator'])
        #~ generator_state_dict = chk_model.generator.state_dict()
        #~ model_state_dict = {k: v for k, v in chk_model.state_dict().items()}
        #~ model.load_state_dict(model_state_dict)
        #~ generator.load_state_dict(generator_state_dict)
        #~ checkpoint['model'][
        #~ print(checkpoint['model'])
        
        checkpoint['model']['generator.net.0.weight'] = checkpoint['generator']['0.weight']
        checkpoint['model']['generator.net.0.bias'] = checkpoint['generator']['0.bias']
        model.load_state_dict(checkpoint['model'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
                
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model.criterion = criterion
        model.load_state_dict(checkpoint['model'])
        opt.start_epoch = checkpoint['epoch'] + 1
        
    model.criterion = criterion
        
    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()

    

    

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)
                
    if opt.tie_weights:
            model.tie_weights()    
    
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
    
    #~ else:
        #~ print('Loading optimizer from checkpoint:')
        #~ optim = checkpoint['optim']
        #~ print(optim)
        #~ # Force change learning rate
        #~ optim.lr = opt.learning_rate
        #~ optim.start_decay_at = opt.start_decay_at
        #~ optim.start_decay = False
    
    if opt.reset_optim or not opt.train_from_state_dict:    
        
            optim = onmt.Optim(
                            opt.optim, opt.learning_rate, opt.max_grad_norm,
                            lr_decay=opt.learning_rate_decay,
                            start_decay_at=opt.start_decay_at
                    )
    
    else:
             print('Loading optimizer from checkpoint:')
             optim = checkpoint['optim']  
             # Force change learning rate
             optim.lr = opt.learning_rate
             optim.start_decay_at = opt.start_decay_at
             optim.start_decay = False
        
    optim.set_parameters(model.parameters())
        
        
        # This doesn't work for me But still there in the main repo 
        # So let's keep it here
    #~ if opt.train_from or opt.train_from_state_dict:
        #~ optim.optimizer.load_state_dict(
            #~ checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    init_buffer(model)
    
    
    bleu_score = eval_translate(model, dicts, opt.valid_src, opt.valid_tgt)
    
    valid_loss = eval(model, validData).data[0]
    valid_ppl = math.exp(min(valid_loss, 100))
    print('* Initial BLEU score : %.2f' % bleu_score)
    print('* Initial Perplexity : %.2f' % valid_ppl)
    print(model)
    print('* Start training ... ')
    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()

#~ def memoryEfficientLoss(outputs, targets, crit, eval=False):
    #~ # compute generations one piece at a time
    #~ loss = 0 
    #~ batch_size = targets.size(1)
    #~ length = len(outputs)
    #~ 
    #~ outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
#~ 
    #~ # outputs is a list of distribution generated by the model
    #~ targets_split = torch.split(targets, 1)
    #~ outputs_split = torch.split(outputs, 1)
    #~ for t, (out_t, target_t) in enumerate(zip(outputs_split, targets_split)):
            #~ 
            #~ if not eval:
                #~ loss_t.div(batch_size).backward()
                #~ 
    #~ grad_output = None if outputs.grad is None else outputs.grad.data
    #~ return loss, grad_output
