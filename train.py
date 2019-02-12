from __future__ import division

import warnings, os
warnings.filterwarnings('ignore')

import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from onmt.trainer.Evaluator import Evaluator
from onmt.trainer.XETrainer import XETrainer
from onmt.trainer.SelfCriticalTrainer import SCSTTrainer
import math
import time

from onmt.trainer.Evaluator import Evaluator
from onmt.ModelConstructor import build_model

#~ from onmt.MultiShardLoader import MultiShardLoader

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-debug', action='store_true',
                    help='Activate debug mode.')
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
parser.add_argument('-adapt_src', default='',
                    help="""source language to adapt""")
parser.add_argument('-adapt_tgt', default='',
                    help="""target language to adapt""")
parser.add_argument('-adapt_id', default=-1,
                    help="""specific language pair ID to adapt""")
parser.add_argument('-override', action='store_true',
                    help="""Overwrite the save file to reduce space consumption""")
parser.add_argument('-loading_strategy', default='all',
                    help="How to load the data. Default strategy = all. Choices=all|piece")
parser.add_argument('-smallest_dataset_multiplicator', metavar='NUMBER', type=int, default=1,
                    help='Use NUMBER times one instance of the smallest dataset in the training of one epoch')
# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_cell', default='lstm',
                    help='Type of LSTM. Support lstm|mlstm')
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
parser.add_argument('-update_every', type=int, default=1,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=2048,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=15,
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
parser.add_argument('-label_smoothing', type=float, default=0.0,
                    help='Applying label smoothing.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-reinforce', action='store_true',
                    help="""Using reinforcement learning""")
parser.add_argument('-reinforce_metrics', default='gleu',
                    help="Type of metrics to use. Options are [gleu|hit].")    
parser.add_argument('-hit_alpha', type=float, default=0.3,
                    help='Coefficient for reward function. The more the focused on rare word rewarding')
parser.add_argument('-critic', default='self',
                    help='Type of critic to be used')
parser.add_argument('-pretrain_critic', action='store_true',
                    help='Type of critic to be used')
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
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-reset_optim', action='store_true',
                    help="""reset the optimization""")
parser.add_argument('-copy_pointer', action='store_true',
                    help="""Pointer network for copying from source""")
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
parser.add_argument('-seed', default=9999, type=int,
                    help="Seed for deterministic runs.")

parser.add_argument('-log_interval', type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument('-save_every', type=int, default=-1,
                    help="Save every this interval.")

# For multilingual configs
parser.add_argument('-share_rnn_enc', action='store_true',
                    help="""Share Rnn Encoder""")
parser.add_argument('-share_rnn_dec', action='store_true',
                    help="""Share Rnn Decoder""")
parser.add_argument('-share_embedding', action='store_true',
                    help="""Share embedding between same language in enc and dec""")
parser.add_argument('-share_projection', action='store_true',
                    help="""Share input and output projection weights of decoder""")
parser.add_argument('-share_attention', action='store_true',
                    help="""Share attentional modules between pair""")
opt = parser.parse_args()

print(opt)

print(torch.__version__)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

torch.manual_seed(opt.seed)



def NMTCriterion(dicts):
    
    crits = dict()
    for i in dicts:
        vocabSize = dicts[i].size()
        
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        if opt.gpus:
            crit.cuda()
        
        crits[i] = crit
    
    return crits



def main():
    
    dataset = dict()
    
    print("Loading dicts from '%s'" % opt.data + "/dicts_info.pt")
    dataset['dicts'] = torch.load(opt.data + "/dicts_info.pt")
    
    pairIDs = list()
    if len(opt.adapt_src) > 0 and len(opt.adapt_tgt) > 0:
    
        # find the source and target ID of the pair we need to adapt
        srcID = dataset['dicts']['srcLangs'].index(opt.adapt_src)
        tgtID = dataset['dicts']['tgtLangs'].index(opt.adapt_tgt)
    
        setIDs = dataset['dicts']['setIDs']
        
        # find the pair ID that we need to adapt
        for i, sid in enumerate(setIDs):
            if sid[0] == srcID and sid[1] == tgtID:
                pairIDs.append(i)
                
        if len(pairIDs) == 0:
            pairIDs = None
    
    else:
        srcID = None
        tgtID = None
        pairIDs = None
    
    # convert string to IDs for easier manipulation
    opt.adapt_src = srcID
    opt.adapt_tgt = tgtID 
    opt.pairIDs = pairIDs
        
    
    dict_checkpoint = opt.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint, map_location=lambda storage, loc: storage)
        #~ dataset['dicts'] = checkpoint['dicts']
    else:
        checkpoint = None
    
    dicts = dataset['dicts']
    
    dataset['valid'] = torch.load(opt.data + "/valid.pt")
    valid_set = dataset['valid']
    
    #~ print("Loading training data from '%s'" % opt.data + "/train.pt.*")
    
    
    
    dataset['train'] = dict()
    
    #~ torch.load(opt.data + "/train.pt.0")
    
    print("Done")
    
    nSets = dicts['nSets']
    setIDs = dicts['setIDs']
    print(' * Vocabulary sizes: ')
    for lang in dicts['langs']:
        print(' * ' + lang + ' = %d' % dicts['vocabs'][lang].size())
        
    # A wrapper to manage data loading
    trainLoader = onmt.MultiShardLoader(opt, dicts)

    trainSets = dict()
    validSets = dict()
    for i in xrange(nSets):
        
        #~ trainSets[i] = onmt.Dataset(dataset['train']['src'][i], dataset['train']['tgt'][i],
                             #~ opt.batch_size, opt.gpus)
            
        validSets[i] = onmt.Dataset(valid_set['src'][i], valid_set['tgt'][i],
                             opt.batch_size, opt.gpus)

        #~ print(' * number of training sentences for set %d: %d' %
          #~ (i, len(dataset['train']['src'][i])))
        

    print('[INFO] * maximum batch size. %d' % opt.batch_size)

    print('[INFO] Building model...')
    
    model, generator = build_model(opt, dicts, nSets)
    
    if opt.train_from_state_dict:
        print('[INFO] Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
              
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if 'critic' not in k}
        checkpoint['critic'] = {k: v for k, v in checkpoint['model'].items() if 'critic' in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(checkpoint['generator'])
        
    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()
    
    model.generator = generator
      

    if not opt.train_from_state_dict :
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

    optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
    )

    nParams = sum([p.nelement() for p in model.parameters()])
    print('[INFO] * number of parameters: %d' % nParams)
    
    
    
    evaluator = Evaluator(model, dataset, opt, cuda=(len(opt.gpus) >= 1))
    
    if opt.reinforce:
        if opt.critic == 'self':
            trainer = SCSTTrainer(model, trainLoader, validSets, dataset, optim, evaluator, opt)
        else:
            from onmt.ModelConstructor import build_critic
            from onmt.trainer.ActorCriticTrainer import A2CTrainer
            critic = build_critic(opt, dicts)
            
            model.critic = critic
            
            trainer = A2CTrainer(model, trainLoader, validSets, dataset, optim, evaluator, opt)
            #~ raise NotImplementedError
    else:
        trainer = XETrainer(model, trainLoader, validSets, dataset, optim, evaluator, opt)
    
    trainer.run(checkpoint=checkpoint)


if __name__ == "__main__":
    main()
