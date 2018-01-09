from __future__ import division

import sys, tempfile
import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import random 
import numpy as np
from onmt.metrics.gleu import sentence_gleu
from onmt.metrics.hit import HitMetrics



def averagePPL(losses, counts):
    
    ppls = []
    
    for i in xrange(len(counts)):
        ppl = math.exp(losses[i] / (counts[i] + 1e-6))
        ppls.append(ppl)
    return sum(ppls) / len(ppls)
    
def compute_score(score, samples, ref, tgtDict, batch_size, average=True):
        
    # probably faster than gpu ?
    sdata = samples.data.cpu().t().tolist()
    rdata = ref.data.cpu().t().tolist()
    
    #~ tgtDict = dicts['tgt']
    
    s = torch.Tensor(batch_size)
    
    for i in xrange(batch_size):
        
        sampledIDs = sdata[i]
        refIDs = rdata[i]
        
        sampledWords = tgtDict.convertToLabels(sampledIDs, onmt.Constants.EOS)
        refWords = tgtDict.convertToLabels(refIDs, onmt.Constants.EOS)
        
        # note: the score function returns a tuple 
        s[i] = score(refWords, sampledWords)[0]
        
    s = s.cuda()
        
    return s


class SCSTTrainer(object):
    
    def __init__(self, model, trainSets, validSets, dataset, optim, evaluator, opt):
        
        self.model = model
        self.trainSets = trainSets
        self.validSets = validSets
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.optim = optim 
        self.evaluator = evaluator
        self.opt = opt
        self.override = opt.override
        
        self.criterions = onmt.Models.NMTCriterion(self.dicts['tgt'], cuda=(len(self.opt.gpus) >= 1))
        
        if opt.reinforce_metrics == 'gleu':
            self.score = sentence_gleu
        elif opt.reinforce_metrics == 'hit':
            hit_scorer = HitMetrics(opt.hit_alpha)
            self.score = hit_scorer.hit
        
        # A flag for language - specific adapting
        self.adapt = False
            
        if opt.adapt_src is not None and opt.adapt_tgt is not None and opt.pairID is not None:
            self.adapt = True
        self.adapt_src = opt.adapt_src
        self.adapt_tgt = opt.adapt_tgt
        self.adapt_pair = opt.pairID
        
        self.best_bleu = 0.00
        
    
    def run(self):
        
        print(self.model)
        self.model.train()
        opt = self.opt
        trainSets = self.trainSets
        validSets = self.validSets
        model = self.model
        dicts = self.dicts
        
        evaluator = self.evaluator
        criterions = self.criterions
        dataset = self.dataset
        optim = self.optim
    
        setIDs = dicts['setIDs']
        
        start_time = time.time()
        
        def trainEpoch(epoch, batchOrder=None):

            # Shuffle mini batch order.
            if not batchOrder:
                batchOrder = dict()
                for i in trainSets:
                    batchOrder[i] = torch.randperm(len(trainSets[i]))

            total_rewards, total_sents = dict(), dict()
            report_rewards, report_tgt_words = dict(), []
            report_tgt_sents = dict()
            report_src_words = []
            start = time.time()
            
            for i in trainSets:
                total_rewards[i] = 0
                total_sents[i] = 0
                report_rewards[i] = 0
                report_tgt_sents[i] = 0
                report_tgt_words.append(0)
                report_src_words.append(0)
            
            dataSizes = [len(trainSets[i]) for i in trainSets]
            
            if self.adapt:
                nSamples = dataSizes[self.adapt_pair]
            else:
                nSamples = sum(dataSizes)
            
            # In order to make sets sample randomly,
            # We create a distribution over the data size
            # In the future we can manipulate this distribution 
            # to create biased sampling when training
            sampleDist = torch.Tensor(len(setIDs))
            iterators = dict()
            for i in xrange(len(setIDs)):
                sampleDist[i] = len(trainSets[i])
                iterators[i] = -1
            sampleDist = sampleDist / torch.sum(sampleDist)

            for i in range(nSamples):
                            
                sampledSet = -1

                if self.adapt:
                    sampledSet = self.adapt_pair
                else:
                    # this loop is very dangerous 
                    # because if the dataset is full then it will loop forever
                    # need a mechanism to halt it
                    while True:
                        # if the sampled set is full then we re-sample 
                        # to ensure that in one epoch we read each example once
                        sampledSet = int(torch.multinomial(sampleDist, 1)[0])
                        if iterators[sampledSet] + 1 < dataSizes[sampledSet]:
                            break
                
                
                iterators[sampledSet] += 1 
                
                tgt_lang = dicts['tgtLangs'][setIDs[sampledSet][1]]
                tgt_dict = self.dicts['vocabs'][tgt_lang]
                
                # Get the batch index from batch order
                batchIdx = batchOrder[sampledSet][iterators[sampledSet]] if epoch > opt.curriculum else iterators[sampledSet]
                
                # Get the batch
                batch = trainSets[sampledSet][batchIdx][:-1]
                batch_size = batch[1].size(1)
                
                # And switch the model to the desired language mode
                model.switchLangID(setIDs[sampledSet][0], setIDs[sampledSet][1])
                model.switchPairID(sampledSet)
                
                # Do forward to the newly created graph
                model.zero_grad()
                
                ref = batch[1][1:]
                batch_size = ref.size(1)
                # Monte-Carlo actions and greedy actions to be sampled
                rl_actions, greedy_actions, logprobs = model(batch, mode='rf')
                
                # reward for samples from stochastic function
                sampled_reward = compute_score(self.score, rl_actions, ref, tgt_dict, batch_size) 
                
                # samples from greedy search
                greedy_reward = compute_score(self.score, greedy_actions, ref, tgt_dict, batch_size) 
                
                # the REINFORCE reward to be the difference between MC and greedy
                rf_rewards = (sampled_reward - greedy_reward)
                
                R = torch.sum(sampled_reward)
                
                total_rewards[sampledSet] += R
                
                # mask: L x B
                seq_mask = rl_actions.data.ne(onmt.Constants.PAD)
                seq_mask = seq_mask.float()
                num_words_sampled = torch.sum(seq_mask)
                
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
                
                # back-prop and compute the gradients
                loss.backward()
       
                # Update the parameters.
                optim.step()

                # Statistics for the current set
                report_rewards[sampledSet] += R
                report_tgt_words[sampledSet] += num_words_sampled
                report_src_words[sampledSet] += batch[0][1].data.sum()
                total_rewards[sampledSet] += R
                total_sents[sampledSet] += batch_size
                report_tgt_sents[sampledSet] += batch_size

                # Logging information
                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    #~ avgTrainLoss = averageReward(report_rewards, report_tgt_words)
                    avgTrainLoss = sum(report_rewards.values()) / sum(report_tgt_sents.values())
                    logOut = ("Epoch %2d, %5d/%5d; ; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed; avg reward: %6.2f; lr: %.6f" %
                                    (epoch, i+1, nSamples,
                                     sum(report_src_words)/(time.time()-start),
                                     sum(report_tgt_words)/(time.time()-start),
                                     time.time()-start_time,
                                     avgTrainLoss,
                                     optim.get_learning_rate()))
                                     
                    for j in xrange(len(setIDs)):
                        
                        report_rewards[j] = 0
                        report_tgt_words[j] = 0
                        report_src_words[j] = 0
                        report_tgt_sents[j] = 0
                        
                    print(logOut)
                    start = time.time()    
                                
                    
                # Saving checkpoints with validation perplexity
                if opt.save_every > 0 and i % opt.save_every == -1 % opt.save_every :
                    valid_bleu_scores = evaluator.eval_translate(validSets)
                    avg_dev_bleu = sum(valid_bleu_scores.values()) / len(valid_bleu_scores)
                    for id in valid_bleu_scores:
                        setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                        print('Validation BLEU Scores for set %s : %g' % (setLangs, valid_bleu_scores[id]))
                    print("Average dev BLEU scores: %g" % avg_dev_bleu)
                    
                    model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                    else model.state_dict())
                    model_state_dict = {k: v for k, v in model_state_dict.items()
                                                            if 'generator' not in k}
                    generator_state_dict = (model.generator.module.state_dict()
                                                                    if len(opt.gpus) > 1
                                                                    else model.generator.state_dict())
                    #  drop a checkpoint

                    ep = float(epoch) - 1.0 + float(i + 1.0) / float(nSamples)

                    checkpoint = {
                            'model': model_state_dict,
                            'generator': generator_state_dict,
                            'dicts': dataset['dicts'],
                            'opt': opt,
                            'epoch': ep,
                            'iteration' : i,
                            'batchOrder' : batchOrder,
                            'optim': optim
                    }
                    
                    
                    
                    if self.override:
                        if self.best_bleu <= avg_dev_bleu:
                            self.best_bleu = avg_dev_bleu
                        file_name='%s.best.pt' % opt.save_model
                        print('Writing to %s' % file_name)
                        torch.save(checkpoint, file_name)
                    else:
                        file_name = '%s_bleu_%.2f_e%.2f.pt' % (opt.save_model, avg_dev_bleu, ep)
                        print('Writing to %s' % file_name)
                        torch.save(checkpoint,  file_name)
                                    
                         
            return 
            #~ return [total_rewards[j] / total_sents[j] for j in xrange(len(setIDs))]
            
        bleu_scores = evaluator.eval_translate(validSets)
        #~ for id in xrange(len(setIDs)):
        for id in bleu_scores:
            setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
            print('Validation BLEU Scores for set %s : %g' % (setLangs, bleu_scores[id]))
        avg_bleu = sum(bleu_scores.values()) / len(bleu_scores)
        print("Average dev BLEU scores: %g" % avg_bleu)
        
        self.best_bleu = avg_bleu
                    
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            trainEpoch(epoch)

            #  (2) evaluate BLEU on the validation set
            valid_bleu_scores = evaluator.eval_translate(validSets)
            avg_dev_bleu = sum(valid_bleu_scores.values()) / len(valid_bleu_scores)
            #~ for i in xrange(len(setIDs)):
            for id in valid_bleu_scores:
                setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                print('Validation BLEU Scores for set %s : %g' % (setLangs, valid_bleu_scores[id]))
            print("Average dev BLEU scores: %g" % avg_dev_bleu)
            
            # learning rate is changed manually - or automatically

            model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                                else model.state_dict())
            model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
                                
            generator_state_dict = (model.generator.module.state_dict()
                                    if len(opt.gpus) > 1
                                    else model.generator.state_dict())
            #  (3) drop a checkpoint
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'iteration' : -1,
                'batchOrder' : None,
                'optim': optim
            }
            
            if self.override:
                if self.best_bleu <= avg_dev_bleu:
                    self.best_bleu = avg_dev_bleu
                file_name='%s.best.pt' % opt.save_model
                print('Writing to %s' % file_name)
                torch.save(checkpoint, file_name)
            else:
                file_name = '%s_bleu_%.2f_e%d.pt' % (opt.save_model, avg_dev_bleu, epoch)
                print('Writing to %s' % file_name)
                torch.save(checkpoint,
                             file_name)
