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
import time, datetime
import random 
import numpy as np

from onmt.Loss import MemoryOptimizedNLLLoss
from onmt.modules import MemoryOptimizedCopyLoss

from onmt.Utils import get_gpu_memory_map


def averagePPL(losses, counts=None):
    
    ppls = []
    
    for i in losses:
        if counts is not None:
            ppl = math.exp(losses[i] / (counts[i] + 1e-6))
        else:
            ppl = losses[i]
        ppls.append(ppl)
        
    return sum(ppls) / len(ppls)


class XETrainer(object):
    
    def __init__(self, model, trainLoader, validSets, dataset, optim, evaluator, opt):
        
        self.model = model
        #~ self.trainSets = trainSets
        self.trainLoader = trainLoader
        self.validSets = validSets
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.optim = optim 
        self.evaluator = evaluator
        self.opt = opt
        self.accumCount = opt.accum_count
        
        if opt.copy_pointer:
            self.criterions = MemoryOptimizedCopyLoss(self.dicts['tgt'], label_smoothing=self.opt.label_smoothing, 
                                                                        shard_size=self.opt.max_generator_batches,
                                                                        cuda=(len(self.opt.gpus) >= 1))
        else:
            self.criterions = MemoryOptimizedNLLLoss(self.dicts['tgt'], label_smoothing=self.opt.label_smoothing, 
                                                                        shard_size=self.opt.max_generator_batches,
                                                                        cuda=(len(self.opt.gpus) >= 1))
        #self.criterions = onmt.Models.NMTCriterion(self.dicts['tgt'], cuda=(len(self.opt.gpus) >= 1))
        # A flag for language - specific adapting
        self.adapt = False
            
        if opt.adapt_src is not None and opt.adapt_tgt is not None and opt.pairIDs is not None:
            self.adapt = True
        self.adapt_src = opt.adapt_src
        self.adapt_tgt = opt.adapt_tgt
        self.adapt_pairs = opt.pairIDs
        
        self.trainLoader.adapt = self.adapt
        self.trainLoader.adapt_src = opt.adapt_src
        self.trainLoader.adapt_src = opt.adapt_tgt
        self.trainLoader.adapt_pairs = opt.pairIDs
        
        self.trainLoader.reset_iterators()
        self.num_updates = 0
        
        self.optim.set_parameters(self.model.parameters())
        
        
    
    def run(self, checkpoint=None):
        
        if checkpoint is not None and not self.opt.reset_optim:
            print("[INFO] Loading optimizer state from checkpoint")
            self.optim.load_state_dict(checkpoint['optim'])
        
        print(self.model)
        self.model.train()
        opt = self.opt
        #~ trainSets = self.trainSets
        trainLoader = self.trainLoader
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
            
            numberGradientsNotAccumulated = 0

            self.trainLoader.reset_iterators()
            batchOrder = self.trainLoader.batchOrder

            total_loss, total_words = dict(), dict()
            report_loss, report_tgt_words = dict(), []
            report_src_words = []
            start = time.time()
            
            for i in validSets:
                total_loss[i] = 0
                total_words[i] = 0
                report_loss[i] = 0
                report_tgt_words.append(0)
                report_src_words.append(0)
                
            nSamples = self.trainLoader.dataSizes()
            
            
            # we don't know the number of samples if not all samples is loaded ?

            #~ for i in range(nSamples):
            while(not trainLoader.finished()):
                
                
                batch, sampledSet = trainLoader.get_batch()
                i = trainLoader.sample_iterator
                
                #~ if sampledSet == -1:
                    #~ break
                
                batch = batch[:-1] # remove the indices
                batch_size = batch[1].size(1)
                
                # And switch the model to the desired language mode
                model.switchLangID(setIDs[sampledSet][0], setIDs[sampledSet][1])
                model.switchPairID(sampledSet)
                
                # Do forward to the newly created graph
                model.zero_grad()
                outputs, attn = model(batch)
                
                # Exclude <s> from targets.
                targets = batch[1][1:]
                # The criterion is for the target language side
                criterion_id = setIDs[sampledSet][1]
                
                loss = criterions.forward(batch, (outputs, attn), targets, criterion_id, 
                                               generator=model.generator, backward=True)
                
                             
                # Update the parameters.
                if numberGradientsNotAccumulated == self.accumCount:
                    optim.step()
                    self.num_updates += self.accumCount
                    numberGradientsNotAccumulated = 0
                else:
                    numberGradientsNotAccumulated += 1

                # Statistics for the current set
                num_words = targets.data.ne(onmt.Constants.PAD).sum()
                report_loss[sampledSet] += loss
                report_tgt_words[sampledSet] += num_words
                report_src_words[sampledSet] += batch[0][1].data.sum()
                total_loss[sampledSet] += loss
                total_words[sampledSet] += num_words

                # Logging information
                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    avgTrainLoss = averagePPL(report_loss, report_tgt_words)
                    
                    logOut = ("Epoch %2d, %5d/%5d; ; %3.0f src tok/s; %3.0f tgt tok/s; ppl: %6.2f; lr: %.6f; %s elapsed" %
                                    (epoch, i+1, nSamples,
                                     sum(report_src_words)/(time.time()-start),
                                     sum(report_tgt_words)/(time.time()-start),
                                     avgTrainLoss,
                                     optim.get_learning_rate(),
                                     str(datetime.timedelta(seconds=int(time.time() - start_time)))))
                                     
                    for j in xrange(len(setIDs)):
                        
                        report_loss[j] = 0
                        report_tgt_words[j] = 0
                        report_src_words[j] = 0
                        
                    print(logOut)
                    start = time.time()    
                                
                    
                # Saving checkpoints with validation perplexity
                if opt.save_every > 0 and i % opt.save_every == -1 % opt.save_every :
                    
                    
                    bleu_scores = evaluator.eval_translate(validSets)
                    for id in bleu_scores:
                        setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                        print('Validation BLEU Scores for set %s : %g' % (setLangs, bleu_scores[id]))

                    valid_ppl = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
                    for id in valid_ppl:
                        setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                        print('Validation perplexity for set %s : %g' % (setLangs, valid_ppl[id]))

                    
                    avgDevPpl = averagePPL(valid_ppl)
                    
                    
                    model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                                        else model.state_dict())
                    model_state_dict = {k: v for k, v in model_state_dict.items()
                                                            if 'generator' not in k}
                    generator_state_dict = (model.generator.module.state_dict()
                                                                    if len(opt.gpus) > 1
                                                                    else model.generator.state_dict())
                    optim_state_dict = optim.state_dict()
                    #  drop a checkpoint

                    checkpoint = {
                            'model': model_state_dict,
                            'generator': generator_state_dict,
                            'dicts': dataset['dicts'],
                            'opt': opt,
                            'num_updates': self.num_updates,
                            'iteration' : i,
                            'batchOrder' : batchOrder,
                            'optim': optim_state_dict
                    }
                    
                    file_name = '%s_ppl_%.2f_e%d.pt'
                    print('Writing to %s_ppl_%.2f_iter%d.pt' % (opt.save_model, avgDevPpl, self.num_updates))
                    torch.save(checkpoint,
                         file_name
                         % (opt.save_model, avgDevPpl, self.num_updates ))

            if numberGradientsNotAccumulated > 0:
                optim.step()

            return [total_loss[j] / (total_words[j]+1e-6) for j in xrange(len(setIDs))]
            
        
        if checkpoint is not None:
            bleu_scores = evaluator.eval_translate(validSets)

            for id in bleu_scores:
                setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                print('Validation BLEU Scores for set %s : %g' % (setLangs, bleu_scores[id]))
            #~ 
            if 'num_updates' in checkpoint:
                self.num_updates = checkpoint['num_updates']
            else:
                self.num_updates = 0
                
            del checkpoint # to save memory
            

        valid_ppl = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)

        for id in valid_ppl:
            setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
            print('Validation perplexity for set %s : %g' % (setLangs, valid_ppl[id]))
            
        
                    
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_losses = trainEpoch(epoch)
            train_ppl = [math.exp(min(train_loss, 100)) for train_loss in train_losses]
            for i in xrange(len(setIDs)):
                print('Training perplexity for set %d : %g' % (i, train_ppl[i]))

            #  (2) evaluate on the validation set
            valid_ppl = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
            avgDevPpl = averagePPL(valid_ppl)
            for id in valid_ppl:
                setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                print('Validation perplexity for set %s : %g' % (setLangs, valid_ppl[id]))
            
            bleu_scores = evaluator.eval_translate(validSets)
            for id in bleu_scores:
                setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                print('Validation BLEU Scores for set %s : %g' % (setLangs, bleu_scores[id]))

            # learning rate is changed manually - or automatically
            if optim.method == 'sgd' and optim.lr_decay < 1. :
                optim.updateLearningRate(avgDevPpl, epoch)

            model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                                else model.state_dict())
            model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
                                
            generator_state_dict = (model.generator.module.state_dict()
                                    if len(opt.gpus) > 1
                                    else model.generator.state_dict())
            optim_state_dict = optim.state_dict()
            #  (3) drop a checkpoint
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'num_updates': self.num_updates,
                'iteration' : -1,
                'batchOrder' : None,
                'optim': optim_state_dict
            }
            
                    
            file_name = '%s_ppl_%.2f_e%d.pt'
            print('Writing to %s_ppl_%.2f_e%d.pt' % (opt.save_model, avgDevPpl, self.num_updates))
            torch.save(checkpoint,
                                         file_name
                                         % (opt.save_model, avgDevPpl, self.num_updates))
