from __future__ import division

import math, os
import torch
from torch.autograd import Variable

import onmt


class MultiShardLoader(object):
    
    def __init__(self, opt, dicts):
        
        self.data_path = opt.data
        
        self.smallest_dataset_multiplicator = opt.smallest_dataset_multiplicator
        
        self.dicts = dicts
        
        nSets = dicts['nSets']
        setIDs = dicts['setIDs']
        self.setIDs = setIDs
        
        # Initialize the place holder for training data
        self.trainSets = dict()
        
        self.trainSets['src'] = list()
        self.trainSets['tgt'] = list()
        
        for i in range(nSets):
            self.trainSets['src'].append(list())
            self.trainSets['tgt'].append(list())
        
        
        self.loading_strategy = opt.loading_strategy
    
        """ First, we have to read the data path to detect n training files"""
        train_files = []
        for root, dirs, files in os.walk(opt.data + "/"):
            for tfile in files:
                if "train." in tfile:
                    train_files.append(tfile)
                        
        self.train_files = train_files
        self.datasets = dict()
        
        if self.loading_strategy == 'all':
            
            """ Load all the data in the shards """ 
            for train_file in sorted(self.train_files, key=lambda a: int(a.split(".")[-1])):
                print("Loading training data from '%s'" % (opt.data + "/" + train_file) + "...")
                data_ = torch.load(opt.data + "/" + train_file)
                
                for i in range(nSets):
                    self.trainSets['src'][i] += data_['src'][i]
                    self.trainSets['tgt'][i] += data_['tgt'][i]
                
                    #~ trainSets[i] = onmt.Dataset(dataset['train']['src'][i], dataset['train']['tgt'][i],
                                         #~ opt.batch_size, opt.gpus)
                                         
            # After loading everything, make a dataset
            for i in range(nSets):
                self.datasets[i] = onmt.Dataset(self.trainSets['src'][i], self.trainSets['tgt'][i],
                                                 opt.batch_size, opt.gpus)
                
            
        
        
        else:
            raise NotImplementedError
            
        
        
        """ Next, if the loading strategy is to load all then pre-load all shards """
    
    def load_shard(self, shard_file):
        pass


    def set_dataset_sizes(self):
        self.sizes = dict()
        
        for i in self.datasets:
            self.sizes[i] = len(self.datasets[i])
            if self.adapt and i not in self.adapt_pairs:
                self.sizes[i] = 0                  
            
        self.smallest_dataset = sorted(((k, v) for k, v in self.sizes.items()), key=lambda x: x[1])[0][0]
        self.sizes[self.smallest_dataset] *= self.smallest_dataset_multiplicator

        
    def reset_iterators(self, batchOrder=None):
        
        self.shard_iterator = None ## for further manipulation with shards
        setIDs = self.setIDs
        # In order to make sets sample randomly,
            # We create a distribution over the data size
            # In the future we can manipulate this distribution 
            # to create biased sampling when training
        sampleDist = torch.Tensor(len(setIDs))
        self.set_iterators = dict()
        self.sample_iterator = -1
        for i in xrange(len(self.dicts['setIDs'])):
            sampleDist[i] = len(self.datasets[i])
            self.set_iterators[i] = -1
            
            if self.adapt:
                if i not in self.adapt_pairs:
                    sampleDist[i] = 0
                    #~ del self.set_iterators[i]
            
        # normalize the distribution 
        self.sampleDist = sampleDist / torch.sum(sampleDist)
        
        #~ print(self.sampleDist)
        
        self.set_dataset_sizes()

        self.nSamples = self.dataSizes()
        
        if batchOrder is None:
            batchOrder = dict()
            for i in self.datasets:
                batchOrder[i] = torch.randperm(self.sizes[i]) if self.sizes[i] > 0 else torch.Tensor()
            self.batchOrder = batchOrder
        else:
            self.batchOrder = batchOrder
            if self.smallest_dataset_multiplicator > 1:
                self.batchOrder[self.smallest_dataset] = torch.cat(
                    [self.batchOrder[self.smallest_dataset]] * self.smallest_dataset_multiplicator, 0)
            elif self.smallest_dataset_multiplicator == 0:
                self.batchOrder[self.smallest_dataset] = torch.Tensor()
        
        #~ self.sizes = [len(self.datasets[i]) for i in self.datasets]
        
            
    
    """ if we scan through the whole dataset """
    def finished(self):
        
        if self.loading_strategy == 'all':
            if self.sample_iterator == self.nSamples -1 :
                return True
            else:
                return False
        else:
            raise NotImplementedError
  
        
    """ return the size of the current shard """
    def dataSizes(self):
        
        #~ sizes = [len(self.datasets[i]) for i in self.datasets]
        #~ 
        #~ if self.adapt:
            #~ nSamples = sizes[self.adapt_pair]
        #~ else:
            #~ nSamples = sum(sizes)
        if not hasattr(self, 'sizes'):
            self.set_dataset_sizes()            
        
        nSamples = sum(self.sizes.values())
        
        return nSamples
        
    def get_batch(self):
        
        
        sampledSet = -1

        #~ if self.adapt:
            #~ sampledSet = self.adapt_pair
        #~ else:
        
        
        # this loop is very dangerous 
        # because if the dataset is full then it will loop forever
        # need a mechanism to halt it
        while True:
            # if the sampled set is full then we re-sample 
            # to ensure that in one epoch we read each example once
            if self.finished():
                break
            
            sampledSet = int(torch.multinomial(self.sampleDist, 1)[0])
            if self.set_iterators[sampledSet] + 1 < self.sizes[sampledSet]:
                break
                
        
        if sampledSet > -1:
        
            # Get the batch index from batch order
            batchIdx = self.batchOrder[sampledSet][self.set_iterators[sampledSet]] 
            
            self.set_iterators[sampledSet] += 1 
            self.sample_iterator += 1
            
            if self.smallest_dataset == sampledSet:
                batch = self.datasets[sampledSet][batchIdx // self.smallest_dataset_multiplicator]
            else:
                batch = self.datasets[sampledSet][batchIdx]
            
            return batch, sampledSet
        else:
            batch = None
            return batch, sampledSet
