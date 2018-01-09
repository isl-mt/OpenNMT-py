from __future__ import division

import sys, tempfile
import onmt
import onmt.modules
#~ from onmt.metrics.gleu import sentence_gleu
#~ from onmt.metrics.sbleu import sentence_bleu
from onmt.metrics.bleu import moses_multi_bleu
#~ from onmt.utils import compute_score
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math

from onmt.metrics.gleu import sentence_gleu
from onmt.metrics.hit import HitMetrics

class Evaluator(object):
    
    def __init__(self, model, dataset, opt, cuda=False):
        
        # some properties
        self.dataset = dataset
        self.dicts = dataset['dicts']
        
        self.setIDs = dataset['dicts']['setIDs']
        
        self.model = model
        
        self.cuda = cuda
        
        self.translator = onmt.InplaceTranslator(self.model, self.dicts, 
                                            beam_size=1, 
                                            cuda=self.cuda)
        self.adapt = False
        
        if opt.adapt_src is not None and opt.adapt_tgt is not None and opt.pairID is not None:
            self.adapt = True
            self.adapt_src = opt.adapt_src
            self.adapt_tgt = opt.adapt_tgt
            self.adapt_pair = opt.pairID
            
        if opt.reinforce_metrics == 'gleu':
            self.score = sentence_gleu
        elif opt.reinforce_metrics == 'hit':
            hit_scorer = HitMetrics(opt.hit_alpha)
            self.score = hit_scorer.hit
        else:
            raise NotImplementedError
            
        
    def setScore(self, score):
        self.score = score
    
    def setCriterion(self, criterion):
        self.criterion = criterion
    
    
    # Compute perplexity of a data given the model
    # For a multilingual dataset, we may need the setIDs of the desired languages
    # data is a dictionary with key = setid and value = DataSet object
    def eval_perplexity(self, data, criterions, setIDs=None):
        
        if setIDs is None:
            setIDs = self.setIDs
            
        model = self.model
        model.eval()
        
        # return a list of losses for each language
        losses = dict()
        
        for sid in data: # sid = setid
            
            if self.adapt:
                # if we are adapting then we only care about that pair
                if sid != self.adapt_pair:
                    continue
            
            dset = data[sid]
            total_loss = 0
            total_words = 0
            
            model.switchLangID(setIDs[sid][0], setIDs[sid][1])
            model.switchPairID(sid)
            
            # each target language requires a criterion, right ?
            criterion = criterions[setIDs[sid][1]]    
            for i in range(len(dset)):
                # exclude original indices
                batch = dset[i][:-1]
                outputs, hiddens = model(batch)
                # exclude <s> from targets
                targets = batch[1][1:]
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)
                loss = criterion(outputs_flat, targets_flat)
                total_loss += loss.data[0]
                total_words += targets.data.ne(onmt.Constants.PAD).sum()
            
            normalized_loss = total_loss / total_words
            losses[sid] = math.exp(min(normalized_loss, 100))
        
        model.train()
        return losses
    
    
    # the only difference of this function and eval translate is that 
    # the metrics is not BLEU, but the scorer
    #~ def eval_reinforce(self, data, verbose=False, setIDs=None):
    #~ 
        #~ score = self.score
        #~ 
        #~ model = self.model
        #~ model.eval()
        #~ 
        #~ # return a list of scores for each language
        #~ total_scores = dict()
        #~ total_sentences = dict()
        #~ 
        #~ total_hit = 0
        #~ total_hit_sentences = 0
        #~ 
        #~ for sid in data: # sid = setid
            #~ if self.adapt:
                #~ # if we are adapting then we only care about that pair
                #~ if sid != self.adapt_pair:
                    #~ continue
                    #~ 
            #~ dset = data[sid]
            #~ model.switchLangID(setIDs[sid][0], setIDs[sid][1])
            #~ model.switchPairID(sid)
            #~ 
            #~ tgt_lang = self.dicts['tgtLangs'][setIDs[sid][1]]
            #~ src_lang = self.dicts['srcLangs'][setIDs[sid][0]]
            #~ tgt_dict = self.dicts['vocabs'][tgt_lang]
            #~ src_dict = self.dicts['vocabs'][src_lang]
            #~ 
            #~ for i in range(len(dset)):
                #~ # exclude original indices
                #~ batch = dset[i][:-1]
                #~ 
                #~ src = batch[0]
                #~ 
                #~ # exclude <s> from targets
                #~ targets = batch[1][1:]
                #~ 
                #~ transposed_targets = targets.data.transpose(0, 1) # bsize x nwords
                #~ 
                #~ pred = self.translator.translate(src)
                #~ 
                #~ batch_size = len(pred)
                #~ 
                #~ for b in range(batch_size)
                
                #~ bpe_string = bpe_token + bpe_token + " "
                #~ 
                #~ for b in range(len(pred)):
                    #~ 
                    #~ ref_tensor = transposed_targets[b].tolist()
                    #~ 
                    #~ decodedSent = tgt_dict.convertToLabels(pred[b], onmt.Constants.EOS)
                    #~ decodedSent = " ".join(decodedSent)
                    #~ decodedSent = decodedSent.replace(bpe_string, '')
                    #~ 
                    #~ refSent = tgt_dict.convertToLabels(ref_tensor, onmt.Constants.EOS)
                    #~ refSent = " ".join(refSent)
                    #~ 
                    #~ refSent = refSent.split('. ; .')[0]
                    #~ 
                    #~ refSent = refSent.replace(bpe_string, '')
                    #~ 
                    #~ 
                    #~ # Flush the pred and reference sentences to temp files 
                    #~ outF.write(decodedSent + "\n")
                    #~ outF.flush()
                    #~ outRef.write(refSent + "\n")
                    #~ outRef.flush()
            
            
        #~ total_gleu = 0
        #~ tgtDict = self.dicts['tgt']
        #~ srcDict = self.dicts['src']
        #~ 
        #~ for i in range(len(data)):
            #~ batch = data[i][:-1]
            #~ src = batch[0]
            #~ ref = batch[1][1:]
            #~ # we need to sample
            #~ sampled_sequence = model.sample(src, max_length=100, argmax=True)
            #~ batch_size = ref.size(1)
            #~ 
            #~ for idx in xrange(batch_size):
            #~ 
                #~ tgtIds = sampled_sequence.data[:,idx]
                #~ 
                #~ tgtWords = tgtDict.convertTensorToLabels(tgtIds, onmt.Constants.EOS)        
                                #~ 
                #~ refIds = ref.data[:,idx]
                #~ 
                #~ refWords = tgtDict.convertTensorToLabels(refIds, onmt.Constants.EOS)
                #~ 
                #~ # return a single score value
                #~ s = score(refWords, tgtWords)
                #~ 
                #~ if len(s) > 2:
                    #~ gleu = s[1]
                    #~ hit = s[2]
                    #~ 
                    #~ if hit >= 0:
                        #~ total_hit_sentences += 1
                        #~ total_hit += hit
                #~ 
                #~ if verbose:
                    #~ sampledSent = " ".join(tgtWords)
                    #~ refSent = " ".join(refWords)
                    #~ 
                    #~ if s[0] > 0:
                        #~ print "SAMPLE :", sampledSent
                        #~ print "   REF :", refSent
                        #~ print "Score =", s
#~ 
                #~ # bleu is scaled by 100, probably because improvement by .01 is hard ?
                #~ total_score += s[0] * 100 
                #~ 
            #~ total_sentences += batch_size
        #~ 
        #~ if total_hit_sentences > 0:
            #~ average_hit = total_hit / total_hit_sentences
            #~ print("Average HIT : %.2f" % (average_hit * 100))
        #~ 
        #~ average_score = total_score / total_sentences
        #~ model.train()
        #~ return average_score
    
    
    # Compute translation quality of a data given the model
    # return: bleu scores (de-facto metrics)
    # and the custom metrics (gleu, hit ... )
    def eval_translate(self, data, beam_size=1, batch_size=16, bpe=True, bpe_token="@"):
        
        model = self.model
        model.eval()
        setIDs = self.setIDs
        
        count = 0

        # one score for each language pair
        bleu_scores = dict()
        
        # return a list of scores for each language
        total_scores = dict()
        total_sentences = dict()
        total_hits = dict()
        total_hit_sentences = dict()
        
        for sid in data: # sid = setid
            
            total_hits[sid] = 0
            total_sentences[sid] = 0
            total_scores[sid] = 0
            total_hit_sentences[sid] = 0
            
            if self.adapt:
                if sid != self.adapt_pair:
                    continue
            
            dset = data[sid]
            model.switchLangID(setIDs[sid][0], setIDs[sid][1])
            model.switchPairID(sid)
            
            tgt_lang = self.dicts['tgtLangs'][setIDs[sid][1]]
            src_lang = self.dicts['srcLangs'][setIDs[sid][0]]
            tgt_dict = self.dicts['vocabs'][tgt_lang]
            src_dict = self.dicts['vocabs'][src_lang]
            
            # we print translations into temp files
            outF = tempfile.NamedTemporaryFile()
            outRef = tempfile.NamedTemporaryFile()
                
            for i in range(len(dset)):
                # exclude original indices
                batch = dset[i][:-1]
                
                src = batch[0]
                
                # exclude <s> from targets
                targets = batch[1][1:]
                
                transposed_targets = targets.data.transpose(0, 1) # bsize x nwords
                
                pred = self.translator.translate(src)
                
                bpe_string = bpe_token + bpe_token + " "
                
                for b in range(len(pred)):
                    
                    ref_tensor = transposed_targets[b].tolist()
                    
                    predWordList = tgt_dict.convertToLabels(pred[b], onmt.Constants.EOS)
                    decodedSent = " ".join(predWordList)
                    decodedSent = decodedSent.replace(bpe_string, '')
                    
                    refWordList = tgt_dict.convertToLabels(ref_tensor, onmt.Constants.EOS)
                    refSent = " ".join(refWordList)
                    
                    refSent = refSent.split('. ; .')[0]
                    
                    refSent = refSent.replace(bpe_string, '')
                    
                    
                    # Flush the pred and reference sentences to temp files 
                    outF.write(decodedSent + "\n")
                    outF.flush()
                    outRef.write(refSent + "\n")
                    outRef.flush()
                    
                    s = self.score(refWordList, predWordList)
                    
                    if len(s) > 2:
                        gleu = s[1]
                        hit = s[2]
                        
                        if hit >= 0:
                            total_hit_sentences[sid] += 1
                            total_hits[sid] += hit
                            
                    total_scores[sid] += s[0] * 100 
                #~ 
                #~ if len(s) > 2:
                    #~ gleu = s[1]
                    #~ hit = s[2]
                    #~ 
                    #~ if hit >= 0:
                        #~ total_hit_sentences += 1
                        #~ total_hit += hit
                #~ 
                #~ if verbose:
                    #~ sampledSent = " ".join(tgtWords)
                    #~ refSent = " ".join(refWords)
                    #~ 
                    #~ if s[0] > 0:
                        #~ print "SAMPLE :", sampledSent
                        #~ print "   REF :", refSent
                        #~ print "Score =", s
#~ 
                #~ # bleu is scaled by 100, probably because improvement by .01 is hard ?
                #~ total_score += s[0] * 100 
            
            total_sentences[sid] += batch_size
                    
            # compute bleu using external script
            bleu = moses_multi_bleu(outF.name, outRef.name)
            outF.close()
            outRef.close()    
            
            bleu_scores[sid] = bleu
            
            #~ if total_hit_sentences > 0:
            #~ average_hit = total_hit / total_hit_sentences
            #~ print("Average HIT : %.2f" % (average_hit * 100))

            #~ average_score = total_score / total_sentences

            # after decoding, switch model back to training mode
            self.model.train()
            
            return bleu_scores
