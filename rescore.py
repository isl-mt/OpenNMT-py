from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
from itertools import repeat

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-src_lang',   default="en",
                    help='Source language')
parser.add_argument('-tgt', required=True,
                    help='True target n-best list (required) in Moses format')
parser.add_argument('-tgt_lang',   default="de",
                    help='Target language')                                      
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-print_nbest', action='store_true',
                    help='Output the n-best list instead of a single sentence')
parser.add_argument('-normalize', action='store_true',
                    help='To normalize the scores based on output length')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    
    # Always pick n_best
        
    rescorer = onmt.Rescorer(opt)

    outF = open(opt.output, 'w')

    goldScoreTotal, goldWordsTotal = 0, 0

    srcBatch, tgtBatch = [], []
    
    tgtScores = []
    tgtWords = []

    sent_count = -1 # counting every source sentence
    global_count = 0 # counting every target sentence (n times more than src)

    tgtF = open(opt.tgt).readlines() 
    srcF = open(opt.src).readlines() 
    
    def run_rescore(hyp_id, srcBatch, tgtBatch, tgtWords, tgtScores):
        
        #~ repeatedSrcBatch = [sent for sent in repeat(sent, 
        
        srcSent = srcBatch[0]
        
        repeatedSrcBatch = [sent for sent in repeat(srcSent, len(tgtBatch))] 
        
        #~ print(repeatedSrcBatch)
        
        scores = rescorer.rescore(repeatedSrcBatch, tgtBatch)
        
        #~ print(len(scores))
        
        #~ print(scores)
        #~ print(srcSent)
    
        for i in xrange(len(scores)):
            
            
            #~ 
            output_line = str(hyp_id) + " ||| " + tgtWords[i] + " ||| " + tgtScores[i] + " " + str(scores[i])
            #~ 
            print(output_line)
    
    
    while True:
        
        tgtLine = tgtF[global_count] 
        
        tgtLine = tgtLine.strip()
        
        # Moses format 
        tgtParts = tgtLine.split(" ||| ")
        
        hyp_id = int(tgtParts[0].strip())
        
        #~ print(hyp_id)
        
        # get a new sentence id
        if hyp_id != sent_count:
            
            if len(srcBatch) > 0:
                
                run_rescore(hyp_id - 1, srcBatch, tgtBatch, tgtWords, tgtScores)
            
            # add data for new sentence 
            srcBatch = []
            tgtBatch = []
            tgtScores = []
            tgtWords = []
            
            
            sent_count += 1 
            srcLine = srcF[sent_count]
            print(srcLine)
            srcTokens = srcLine.strip().split()
            srcBatch += [srcTokens]
            
        tgtSent = tgtParts[1].strip()
        
        
        
        tgtTokens = tgtSent.split()  
        tgtBatch += [tgtTokens]
        tgtWords += [tgtSent]
        tgtScores += [tgtParts[2]]
        
        global_count += 1  
        
        # end of line
        if global_count >= len(tgtF):
            
            if len(srcBatch) > 0:
                run_rescore(hyp_id, srcBatch, tgtBatch, tgtWords, tgtScores)
            
            break
        
    
    # Read data from input file
    # Input file format:
    # ID ||| sentence ||| scores
    
    #~ for line in addone(open(opt.src)):
        #~ if line is not None:
            #~ 
            #~ 
            #~ srcTokens = line.split()
            #~ srcBatch += [srcTokens]
            #~ 
            #~ for i in xrange(opt.batch_size):
                #~ 
            #~ if tgtF:
                #~ tgtTokens = tgtF.readline().split() if tgtF else None
                #~ tgtBatch += [tgtTokens]
#~ 
            #~ if len(srcBatch) < opt.batch_size:
                #~ continue
        #~ else:
            #~ # at the end of file, check last batch
            #~ if len(srcBatch) == 0:
                #~ break
#~ 
        #~ predBatch, predScore, goldScore = translator.translate(srcBatch,
                                                               #~ tgtBatch)
        #~ 
        #~ if opt.normalize:
            #~ predBatch_ = []
            #~ predScore_ = []
            #~ for bb, ss in zip(predBatch, predScore):
                    #~ ss_ = [s_/numpy.maximum(1., len(b_)) for b_,s_ in zip(bb,ss)]
                    #~ sidx = numpy.argsort(ss_)[::-1]
                    #~ predBatch_.append([bb[s] for s in sidx])
                    #~ predScore_.append([ss_[s] for s in sidx])
            #~ predBatch = predBatch_
            #~ predScore = predScore_
                                                              #~ 
        #~ predScoreTotal += sum(score[0] for score in predScore)
        #~ predWordsTotal += sum(len(x[0]) for x in predBatch)
        #~ if tgtF is not None:
            #~ goldScoreTotal += sum(goldScore)
            #~ goldWordsTotal += sum(len(x) for x in tgtBatch)
            #~ 
#~ 
        #~ for b in range(len(predBatch)):
            #~ # Pred Batch always have n-best outputs  
            
            #~ count += 1
            #~ # Best sentence = having highest log prob
#~ 
            #~ if not opt.print_nbest:
                #~ outF.write(" ".join(predBatch[b][0]) + '\n')
                #~ outF.flush()
#~ 
            #~ if opt.verbose:
                #~ srcSent = ' '.join(srcBatch[b])
                #~ if translator.tgt_dict.lower:
                    #~ srcSent = srcSent.lower()
                #~ print('SENT %d: %s' % (count, srcSent))
                #~ print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                #~ print("PRED SCORE: %.4f" % predScore[b][0])
#~ 
                #~ if tgtF is not None:
                    #~ tgtSent = ' '.join(tgtBatch[b])
                    #~ if translator.tgt_dict.lower:
                        #~ tgtSent = tgtSent.lower()
                    #~ print('GOLD %d: %s ' % (count, tgtSent))
                    #~ print("GOLD SCORE: %.4f" % goldScore[b])
#~ 
                #~ if opt.print_nbest :
                    #~ print('\nBEST HYP:')
                    #~ for n in range(opt.n_best):
                        #~ idx = sorted_index[n]
                        #~ print("%d ||| %s %.6f" % (count, " ".join(predBatch[b][idx], predScore[b][idx])))
                        #~ outF.write("%d ||| %s %.6f\n" % (count, " ".join(predBatch[b][idx], predScore[b][idx])))
                        #~ outF.flush()
                        #~ 
                                             #~ 
#~ 
                #~ print('')
#~ 
        #~ srcBatch, tgtBatch = [], []
#~ 
    #~ reportScore('PRED', predScoreTotal, predWordsTotal)
    #~ if tgtF:
        #~ reportScore('GOLD', goldScoreTotal, goldWordsTotal)
#~ 
    #~ if tgtF:
        #~ tgtF.close()
#~ 
    #~ if opt.dump_beam:
        #~ json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))


if __name__ == "__main__":
    main()
