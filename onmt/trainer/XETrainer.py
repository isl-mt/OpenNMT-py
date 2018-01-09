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


def averagePPL(losses, counts):
    
    ppls = []
    
    for i in xrange(len(counts)):
        ppl = math.exp(losses[i] / (counts[i] + 1e-6))
        ppls.append(ppl)
    return sum(ppls) / len(ppls)


class XETrainer(object):
    
    def __init__(self, model, trainSets, validSets, dataset, optim, evaluator, opt):
        
        self.model = model
        self.trainSets = trainSets
        self.validSets = validSets
        self.dicts = dataset['dicts']
        self.dataset = dataset
        self.optim = optim 
        self.evaluator = evaluator
        self.opt = opt
        
        self.criterions = onmt.Models.NMTCriterion(self.dicts['tgt'], cuda=(len(self.opt.gpus) >= 1))
        
        # A flag for language - specific adapting
        self.adapt = False
            
        if opt.adapt_src is not None and opt.adapt_tgt is not None and opt.pairID is not None:
            self.adapt = True
        self.adapt_src = opt.adapt_src
        self.adapt_tgt = opt.adapt_tgt
        self.adapt_pair = opt.pairID
        
    
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

            total_loss, total_words = dict(), dict()
            report_loss, report_tgt_words = dict(), []
            report_src_words = []
            start = time.time()
            
            for i in trainSets:
                total_loss[i] = 0
                total_words[i] = 0
                report_loss[i] = 0
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
                outputs, hiddens = model(batch)
                
                # Exclude <s> from targets.
                targets = batch[1][1:]
                # The criterion is for the target language side
                criterion = criterions[setIDs[sampledSet][1]]
                
                loss_v = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                loss = loss_v.data[0]
                
                loss_v.div(batch_size).backward()
                             
                # Update the parameters.
                optim.step()

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
                    logOut = ("Epoch %2d, %5d/%5d; ; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed; ppl: %6.2f; lr: %.6f" %
                                    (epoch, i+1, nSamples,
                                     sum(report_src_words)/(time.time()-start),
                                     sum(report_tgt_words)/(time.time()-start),
                                     time.time()-start_time,
                                     avgTrainLoss,
                                     optim.get_learning_rate()))
                                     
                    for j in xrange(len(setIDs)):
                        
                        report_loss[j] = 0
                        report_tgt_words[j] = 0
                        report_src_words[j] = 0
                        
                    print(logOut)
                    start = time.time()    
                                
                    
                # Saving checkpoints with validation perplexity
                if opt.save_every > 0 and i % opt.save_every == -1 % opt.save_every :
                    valid_losses = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
                    valid_ppl = [math.exp(min(valid_loss, 100)) for valid_loss in valid_losses]
                    #~ valid_ppl = " ".join([str(math.exp(min(valid_loss, 100))) for valid_loss in valid_losses])
                    for i in xrange(len(setIDs)):
                        setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][i])
                        print('Validation perplexity for set %s : %g' % (setLangs, valid_ppl[i]))
                    
                    
                    avgDevPpl = sum(valid_ppl) / len(valid_ppl)
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
                    
                    file_name = '%s_ppl_%.2f_e%.2f.pt'
                    #~ valid_ppl = "_".join([("%.2f" % math.exp(min(valid_loss, 100))) for valid_loss in valid_losses])
                    print('Writing to %s_ppl_%.2f_e%.2f.pt' % (opt.save_model, avgDevPpl, ep))
                    torch.save(checkpoint,
                         file_name
                         % (opt.save_model, avgDevPpl, ep))
            return [total_loss[j] / total_words[j] for j in xrange(len(setIDs))]
            
        #~ valid_losses = eval(model, criterions, validSets, setIDs)
        bleu_scores = evaluator.eval_translate(validSets)
        for i in xrange(len(setIDs)):
            setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][i])
            print('Validation BLEU Scores for set %s : %g' % (setLangs, bleu_scores[i]))
        
        valid_losses = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
        
        #~ for i in xrange(len(setIDs)):
        for id in valid_losses:
            setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
            print('Validation perplexity for set %s : %g' % (setLangs, valid_losses[id]))
            
        
                    
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_losses = trainEpoch(epoch)
            train_ppl = [math.exp(min(train_loss, 100)) for train_loss in train_losses]
            for i in xrange(len(setIDs)):
                print('Training perplexity for set %d : %g' % (i, train_ppl[i]))

            #  (2) evaluate on the validation set
            valid_ppl = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
            avgDevPpl = sum(valid_ppl) / len(valid_ppl)
            for id in valid_losses:
                setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][id])
                print('Validation perplexity for set %s : %g' % (setLangs, valid_losses[id]))
            
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
            
                    
            file_name = '%s_ppl_%.2f_e%d.pt'
            print('Writing to %s_ppl_%.2f_e%d.pt' % (opt.save_model, avgDevPpl, epoch))
            torch.save(checkpoint,
                                         file_name
                                         % (opt.save_model, avgDevPpl, epoch))
