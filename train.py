from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
from onmt.metrics.gleu import sentence_gleu
from onmt.metrics.sbleu import sentence_bleu
from onmt.metrics.bleu import moses_multi_bleu
from onmt.metrics.hit import HitMetrics
from onmt.trainer.Evaluator import Evaluator
from onmt.utils import split_batch, compute_score
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import random 
import numpy as np



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
parser.add_argument('-balance_batch', default=1, type=int,
                    help="""balance mini batches (same source sentence length)""")
parser.add_argument('-join_vocab', action='store_true',
                    help='Use a bidirectional encoder')
# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=512,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=512,
                    help='Word embedding sizes')
parser.add_argument('-hidden_output_size', type=int, default=-1,
                    help='Size of the final hidden (output embedding)')
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
parser.add_argument('-computational_batch_size', type=int, default=-1,
                    help='Maximum batch size for computation. By default it is the same as batch size. But we can split the large minibatch to fit in the GPU.')
parser.add_argument('-max_generator_batches', type=int, default=64,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")                   
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
parser.add_argument('-word_dropout', type=float, default=0.1,
                    help='Dropout probability; applied on discrete word types.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
                    
# for reinforcement learning
parser.add_argument('-reinforce_rate', type=float, default=0.0,
                    help='Rate of using reinforcement learning during training')
parser.add_argument('-hit_alpha', type=float, default=0.5,
                    help='Rate of balancing gleu and hit')
parser.add_argument('-reinforce_metrics', default='gleu',
                    help='Metrics for reinforcement learning. Default = gleu')
parser.add_argument('-reinforce_sampling_number', type=int, default=1,
                    help='Number of samples during reinforcement learning')
parser.add_argument('-actor_critic', action='store_true',
                    help='Use actor critic algorithm (default is self-critical)')             
parser.add_argument('-normalize_rewards', action='store_true',
                    help='Normalize the rewards')     
parser.add_argument('-pretrain_critic', action='store_true',
                    help='Pretrain the critic')  
parser.add_argument('-use_entropy', action='store_true',
                    help='Use the entropy term in the A2C loss function)')             
parser.add_argument('-entropy_coeff', type=float, default=0.01,
                    help='Entropy coefficient in the A2C loss function.') 
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
parser.add_argument('-disable_cudnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-seed', default=9999, nargs='+', type=int,
                    help="Seed for deterministic runs.")

parser.add_argument('-log_interval', type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument('-save_every', type=int, default=-1,
                    help="Save every this interval.")
parser.add_argument('-sample_every', type=int, default=1e99,
                    help="Save every this interval.")

parser.add_argument('-valid_src', default='',
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', default='',
                    help="Path to the validation target data")                   
opt = parser.parse_args()

if opt.computational_batch_size <= 0 :
    opt.computational_batch_size = opt.batch_size

print(opt)



print '__PYTORCH VERSION:', torch.__version__


if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])
    cuda.manual_seed_all(opt.seed)
    
    if not opt.disable_cudnn:
        print '__CUDNN VERSION:', torch.backends.cudnn.version()
    else:
        torch.backends.cudnn.enabled = False
    
    
    
    #~ device_name = cuda.get_device_name(opt.gpus[0])
    #~ print("Using CUDA on %s" % device_name)


torch.manual_seed(opt.seed)

# METRICS for REINFORCE
if opt.reinforce_metrics == 'gleu':
    score = sentence_gleu
elif opt.reinforce_metrics == 'sbleu':
    score = sentence_bleu
elif opt.reinforce_metrics == 'hit':
    scorer = HitMetrics(opt.hit_alpha)
    score = scorer.hit

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit

 
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
    
    sampled_sequence = model.sample((variable, length), argmax=False)
    
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
        
        s = score(refWords, tgtWords)[0]
        
        print "Score =", s
    
    if eval:
        model.train()
        
    print("\n")
    



def trainModel(model, trainData, validData, dataset, optims, criterion, critic):
    
    model.train()
    
    optim = optims[0]
    
    if critic is not None:
        critic_optim = optims[1]
        critic.train()
    
    dicts = dataset['dicts']

    start_time = time.time()

    def trainEpoch(epoch, batchOrder=None):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        
        if not batchOrder:
            batchOrder = torch.randperm(len(trainData))

        stats = onmt.Stats(optim)
        nSamples = len(trainData)
        for i in range(nSamples):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            batch = trainData[batchIdx][:-1]
            batch_size = batch[1].size(1)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            
            model.zero_grad()

            train_mode = 'xe'
            # let's stick to one mode for now ?
            if 0 < opt.reinforce_rate:
                train_mode = 'rf' # default reinforce mode = self critical
                if opt.actor_critic: 
                    train_mode = 'ac'
            
            # switch mode: display necessary information based on 
            # XE or RF
            stats.switch_mode(train_mode)
                        
            # For Cross Entropy mode training        
            if train_mode == 'xe':
                
                outputs , _ = model(batch, mode=train_mode)               
                split_targets = batch[1][1:]
                
                flat_outputs = outputs.view(-1, outputs.size(-1))
                flat_targets = split_targets.view(-1)
                
                # Loss is computed by nll criterion
                loss = criterion(flat_outputs, flat_targets)
                loss_value = loss.data[0]
                
                norm_value = batch_size
            
                if opt.optim == 'yellowfin':
                    norm_value = num_words
                
                loss.div(norm_value).backward()

                loss_xe = loss_value
                
                stats.report_loss_xe += loss_xe
                
                
                stats.total_loss_xe += loss_xe
                stats.total_words_xe += num_words
                            
            # For reinforcement learning mode
            elif train_mode == 'rf':
                ref = batch[1][1:]
                batch_size = ref.size(1)
                # Monte-Carlo actions and greedy actions to be sampled
                rl_actions, greedy_actions, logprobs, entropies = model(batch, mode=train_mode, gen_entropy=opt.use_entropy)
                
                # reward for samples from stochastic function
                sampled_reward = compute_score(score, rl_actions, ref, dicts, batch_size) 
                
                stats.total_sent_reward += torch.sum(sampled_reward)
                
                # mask: L x B
                seq_mask = rl_actions.data.ne(onmt.Constants.PAD)
                seq_mask = seq_mask.float()
                num_words_sampled = torch.sum(seq_mask)
                stats.report_sampled_words += num_words_sampled
                
                # samples from greedy search
                greedy_reward = compute_score(score, greedy_actions, ref, dicts, batch_size) 
                
                # the REINFORCE reward to be the difference between MC and greedy
                rf_rewards = (sampled_reward - greedy_reward)
                
                # centralize the rewards to make learning faster ?
                # it can cause a GPU error when batch size < 100 (probably numerical error with float32)
                # and it doesn't seem to be very helpful for this case
                if opt.normalize_rewards:  
                    rf_rewards = (rf_rewards - rf_rewards.mean()) / (rf_rewards.std() + np.finfo(np.float32).eps)
                                
                # Reward cumulative backward:
                length = rl_actions.size(0)
                weight_variable = Variable(seq_mask, requires_grad=False)
                expanded_reward = rf_rewards.unsqueeze(0).expand_as(seq_mask)
                reward_variable = Variable(expanded_reward, requires_grad=False)
                
                # REINFORCE loss (Sutton et al. 1992)
                action_loss = -(logprobs * reward_variable * weight_variable).sum()
                
                # normalize the loss by batch size
                action_loss.div(batch_size)
                
                loss = action_loss
                
                # Use another entropy term in A2C
                if opt.use_entropy:
                    entropy_loss = torch.sum(entropies * weight_variable)
                    loss = loss - opt.entropy_coeff * entropy_loss.div(num_words_sampled)
                
                # back-prop and compute the gradients
                loss.backward()
                
                # old-style with PyTorch provided reinforce function
                # deprecated due to lack of efficiency and black-box as well
                # will need to rewrite the actor critic as well
                #~ for t in xrange(length):
                    #~ reward_t = rf_rewards.clone()
                    #~ mask_t = seq_mask[t]
                    #~ reward_t = torch.mul(reward_t , mask_t)
                    #~ model.saved_actions[t].reinforce(reward_t.unsqueeze(0).t())
                
                #~ torch.autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
                
                del greedy_reward
                del greedy_actions
                
                
                # Update the parameters and free the action list.    
                for action in model.saved_actions:
                    del action
                model.saved_actions = []
            
            elif train_mode == "ac":
                critic.zero_grad()
                
                ref = batch[1][1:]
                batch_size = ref.size(1)
                
                # Monte-Carlo actions and greedy actions to be sampled
                if opt.pretrain_critic:
                    src = batch[0]
                    variable = Variable(src[0].data, volatile=True)
                    length = Variable(src[1].data, volatile=True)
                    
                    rl_actions  = model.sample((variable, length), argmax=False)
                else:
                    rl_actions  = model(batch, mode=train_mode)
                    
                length = rl_actions.size(0)
                
                # feed the actions into the critic, to predict the state values (V):
                critic_input = Variable(rl_actions.data, requires_grad=False)
                critic_output = critic(batch[0], critic_input)
                
                # mask: L x B
                seq_mask = rl_actions.data.ne(onmt.Constants.PAD).float()
                num_words_sampled = torch.sum(seq_mask)
                stats.report_sampled_words += num_words_sampled
                
                # reward for samples from stochastic function
                sampled_reward = compute_score(score, rl_actions, ref, dicts, batch_size) 
                stats.total_sent_reward += torch.sum(sampled_reward)
                
                # compute loss for the critic
                # first we have to expand the reward
                expanded_reward = sampled_reward.unsqueeze(0).expand_as(seq_mask)
                
                #~ print(expanded_reward)
                #~ print(seq_mask)
                
                # compute weighted loss for critic
                #~ reward_seq = torch.FloatTensor([sampled_reward] * length)
                #~ if opt.
                #~ reward_variable = Variable(reward_seq).contiguous()
                reward_variable = Variable(expanded_reward)
                weight_variable = Variable(seq_mask)
                #~ critic_loss = onmt.modules.Loss.weighted_mse_loss(critic_output, reward_variable, weight_variable)
                
                # mean squared error loss
                advantage = weight_variable * (critic_output - reward_variable) 
                critic_loss = torch.sum(advantage.pow(2))
                
                stats.total_critic_loss += critic_loss.data[0]
                
                
                # backward for critic model
                critic_loss.div(num_words_sampled).backward()
                
                critic_optim.step()
                
                if not opt.pretrain_critic:
                    # update the actor (nmt model)
                    model_reward = -advantage.data
                    model_reward.div(batch_size) # important: normalize by batch-size 
                    
                    reward_steps = model_reward.chunk(length, dim=0)
                    #~ print(model_reward)
                    length = rl_actions.size(0)
                    for t in xrange(length):
                        model.saved_actions[t].reinforce(reward_steps[t].t())
                        
                    # We backward from stochastic actions, ignoring the gradOutputs
                    torch.autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
                    
                    # Update the parameters and free the action list.    
                    model.saved_actions = []
                    
                                    
            stats.report_tgt_words_xe += num_words
            stats.report_sentences += batch_size
            
            #~ _train_f(batch)
            
            # We split the minibatch into smaller ones when we cannot fit in the GPU
            #~ splitted_batches = split_batch(batch, opt.computational_batch_size)
                                           
            
            #~ train_mode = _train_f(batch, batch_size, stats)
            #~ for mini_batch in splitted_batches:
                #~ # Do forward on the mini batch 
                #~ # And get the gradients
                #~ _train_f(mini_batch, batch_size, stats)
                #~ 
                #~ # store the gradients in the buffer  
                #~ gBuffer.add_buffer()
                #~ 
                #~ # zero grad the model for the next mini batch
                #~ model.zero_grad()
                #~ del mini_batch
            
            # transfer the gradients from buffer to model
            #~ gBuffer.accumulate_buffer()
            
            # finally, do the SGD update
            
            if train_mode != 'ac' or opt.pretrain_critic == False:
                optim.step()            
                        
            stats.report_src_words += batch[0][1].data.sum()
            stats.report_tgt_words += num_words

            # Sampling (experimental):
            if i % opt.sample_every == -1 % opt.sample_every and opt.sample_every > 0:
                sample(model, batch, dicts)

            
            if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                
                stats.log(i, epoch, len(trainData))
                stats.reset_stats()

            if opt.save_every > 0 and i % opt.save_every == -1 % opt.save_every :
                valid_loss = evaluator.eval_perplexity(validData, criterion)
                valid_ppl = math.exp(min(valid_loss, 100))
                valid_bleu = evaluator.eval_translate(batch_size = opt.eval_batch_size)
                valid_score = evaluator.eval_reinforce(validData, score)
                print('Validation perplexity: %g' % valid_ppl)
                print('Validation BLEU: %.2f' % valid_bleu)
                print('Validation score: %.2f' % valid_score)
                
                if critic is not None:
                    valid_critic_loss = evaluator.eval_critic(validData, dicts, score)
                    print('* Valid Critic Loss : %.4f' % valid_critic_loss)
                
                model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                else model.state_dict())
                #~ model_state_dict = {k: v for k, v in model_state_dict.items()}
                
                if critic is not None:
                    critic_state_dict = critic.state_dict()
                else:
                    critic_state_dict = None
                
                #  drop a checkpoint
                ep = float(epoch) - 1 + (i + 1) / nSamples
                checkpoint = {
                        'model': model_state_dict,
                        'dicts': dataset['dicts'],
                        'opt': opt,
                        'epoch': ep,
                        'iteration' : i,
                        'batchOrder' : batchOrder,
                        'optim': optim,
                        'critic': critic_state_dict
                }
                
                file_name = '%s_ppl_%.2f_score_%.2f_bleu_%.2f_e%.2f.pt'
                print('Writing to ' + file_name % (opt.save_model, valid_ppl, valid_bleu, valid_score, ep))
                torch.save(checkpoint,
                                     file_name
                                     % (opt.save_model, valid_ppl, valid_bleu, valid_score, ep))
        return stats.total_loss_xe / (stats.total_words_xe + 1e-6)

    for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
        print('')

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)

        #  (2) evaluate on the validation set
        valid_loss = evaluator.eval_perplexity(validData, criterion)
        valid_ppl = math.exp(min(valid_loss, 100))
        valid_bleu = evaluator.eval_translate(batch_size = opt.eval_batch_size)
        valid_score = evaluator.eval_reinforce(validData, score)
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation BLEU: %.2f' % valid_bleu)
        print('Validation score: %.2f' % valid_score)
        
        if critic is not None:
            valid_critic_loss = evaluator.eval_critic(validData, dicts, score)
            print('* Valid Critic Loss : %.4f' % valid_critic_loss)

        #  (3) update the learning rate
        #~ optim.updateLearningRate(valid_ppl, epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()}
        
        if critic is not None:
            critic_state_dict = critic.state_dict()
            critic_state_dict = {k: v for k, v in critic_state_dict.items()}
        else:
            critic_state_dict = None

        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'iteration' : -1,
            'batchOrder' : None,
            'optim': optim,
            'critic': critic_state_dict
        }
        
                
        file_name = '%s_ppl_%.2f_score_%.2f_bleu_%.2f_e%d.pt'
        print('Writing to ' + file_name % (opt.save_model, valid_ppl, valid_score, valid_bleu, epoch))
        torch.save(checkpoint,
                   file_name % (opt.save_model, valid_ppl, valid_score, valid_bleu, epoch))


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
                             data_type=dataset.get("type", "text"), balance=(opt.balance_batch==1))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.eval_batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')


    criterion = NMTCriterion(dataset['dicts']['tgt'].size())
    
    embeddings = onmt.utils.createEmbeddings(opt, dicts)

    model = onmt.utils.createNMT(opt, dicts, embeddings)
    print "Neural Machine Translation Model"
    print(model)
    
    # by default: no critic
    critic = None
    
    if opt.actor_critic:
        critic_embeddings = onmt.utils.createEmbeddings(opt, dicts)
        
        critic = onmt.utils.createCritic(opt, dicts, critic_embeddings)
        print "Created Critic Model"
        print(critic)
    

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        checkpoint['model']['generator.net.0.weight'] = checkpoint['generator']['0.weight']
        checkpoint['model']['generator.net.0.bias'] = checkpoint['generator']['0.bias']
        model.load_state_dict(checkpoint['model'])
        opt.start_epoch = checkpoint['epoch'] + 1
    
    
    critic_loaded_flag = False
    if opt.train_from_state_dict:
                
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if 'criterion' not in k}
        model.load_state_dict(model_state_dict)
        opt.start_epoch = int(math.floor(checkpoint['epoch'] + 1))
        del checkpoint['model'] 
        
        if critic is not None and 'critic' in checkpoint:
            if checkpoint['critic'] is not None:
                print('Loading critic weights from checkpoint')
                critic.load_state_dict(checkpoint['critic'])
                critic_loaded_flag = True
        

    if len(opt.gpus) >= 1:
        model.cuda()
        if critic is not None:
            critic.cuda()
    else:
        model.cpu()
        if critic is not None:
            critic.cpu()

    if not opt.train_from_state_dict and not opt.train_from:
        # initialize parameters for the nmt model
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        model.encoder.load_pretrained_vectors(opt)
        model.decoder.load_pretrained_vectors(opt)
        
    # initialize parameters for the critic
    if critic is not None and critic_loaded_flag == False:
        for p in critic.parameters():
            p.data.uniform_(-0.01, 0.01)
                
    if opt.tie_weights:
        print("Share weights between decoder input and output embeddings")
        model.tie_weights()   
        
    if opt.join_vocab:
        print("Share weights between source and target embeddings")
        model.tie_join_embeddings()
    
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
    
    
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
         del checkpoint['optim']
        
    optim.set_parameters(model.parameters())
    optim.setLearningRate(opt.learning_rate)
    
    # Separate Optimizer for the critic 
    if critic is not None:
        critic_optim = onmt.Optim(
                opt.optim, opt.learning_rate, opt.max_grad_norm,
                lr_decay=opt.learning_rate_decay,
                start_decay_at=opt.start_decay_at
        )
        critic_optim.set_parameters(critic.parameters())
    else:
        critic_optim = None
        
        # This doesn't work for me But still there in the main repo 
        # So let's keep it here
    #~ if opt.train_from or opt.train_from_state_dict:
        #~ optim.optimizer.load_state_dict(
            #~ checkpoint['optim'].optimizer.state_dict())
            
    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    
    global evaluator
    evaluator = Evaluator(model, critic, dataset, opt.valid_src, opt.valid_tgt, cuda=(len(opt.gpus) >= 1))

    #~ valid_loss = evaluator.eval_perplexity(validData, criterion)
    #~ valid_ppl = math.exp(min(valid_loss, 100))
    
    bleu_score = evaluator.eval_translate(batch_size = opt.eval_batch_size)
    print('* Initial BLEU score : %.2f' % bleu_score)
    valid_score = evaluator.eval_reinforce(validData, score, verbose=False) # verbose=True just for debugging
    print('* Initial RF Score : %.2f' % valid_score) 
    #~ print('* Initial Perplexity : %.2f' % valid_ppl)
    
    if critic is not None:
        valid_critic_loss = evaluator.eval_critic(validData, dicts, score)
        print('* Valid Critic Loss : %.4f' % valid_critic_loss)
    
    print('* Start training ... ')
    
    if opt.reinforce_rate >= 1.0 and opt.actor_critic and opt.pretrain_critic:
        print('* Pretraining the critic ... ')
    trainModel(model, trainData, validData, dataset, (optim, critic_optim), criterion, critic)


if __name__ == "__main__":
    main()

