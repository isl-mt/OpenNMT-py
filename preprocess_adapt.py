import onmt
import onmt.Markdown
import argparse
import torch
import os.path
from collections import OrderedDict


parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-load_from', required=True, type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                    help="Path to the validation target data")
parser.add_argument('-src_langs', required=True,
                    help="Path to the validation target data")
parser.add_argument('-tgt_langs', required=True,
                    help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
#~ parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    #~ help="Size of the target vocabulary")
parser.add_argument('-vocab',
                    help="The prefix to vocab file, will be concatenated with the language. \
                                                  For example: vocab.en")
#~ parser.add_argument('-tgt_vocab',
                    #~ help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=50,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=50,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def makeVocabulary(filenames, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)
                      
    for filename in filenames:
            print("Reading file " + filename)
            with open(filename) as f:
                for sent in f.readlines():
                    for word in sent.split():
                        vocab.add(word)
    #~ with open(filename) as f:
        #~ for sent in f.readlines():
            #~ for word in sent.split():
                #~ vocab.add(word)
#~ 
    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        vocabFile = vocabFile + "." + name
        if os.path.isfile(vocabFile):
        # If given, load existing word dictionary.
            print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
            vocab = onmt.Dict()
            vocab.loadFile(vocabFile)
            print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)
        vocab = genWordVocab

    print("Done")
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcWords = sline.split()
        tgtWords = tline.split()

        if len(srcWords) <= opt.src_seq_length \
           and len(tgtWords) <= opt.tgt_seq_length:

            # Check truncation condition.
            if opt.src_seq_length_trunc != 0:
                srcWords = srcWords[:opt.src_seq_length_trunc]
            if opt.tgt_seq_length_trunc != 0:
                tgtWords = tgtWords[:opt.tgt_seq_length_trunc]

            if opt.src_type == "text":
                src += [srcDicts.convertToIdx(srcWords,
                                              onmt.Constants.UNK_WORD)]
            elif opt.src_type == "img":
                loadImageLibs()
                src += [transforms.ToTensor()(
                    Image.open(opt.src_img_dir + "/" + srcWords[0]))]

            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)]
            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print(('Prepared %d sentences ' +
          '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))

    return src, tgt


def main():
    
    print("Loading pre-processed data from " + opt.load_from)
    dataset = torch.load(opt.load_from)
    
    dicts = dataset['dicts']
    
    srcLangs = opt.src_langs.split("|")
    tgtLangs = opt.tgt_langs.split("|")
    
    srcFiles = opt.train_src.split("|")
    tgtFiles = opt.train_tgt.split("|")

    validSrcFiles = opt.valid_src.split("|")
    validTgtFiles = opt.valid_tgt.split("|")
    
    langs = dicts['langs']
    
    nPairs = len(srcLangs)
    
    print(dicts['setIDs'])
    
    train = {}
    train['src'] = list()
    train['tgt'] = list()
    dicts['setIDs'] = list()
    dicts['setLangs'] = list()
    
    valid = {}
    valid['src'] = list()
    valid['tgt'] = list()
    
    for i in range(nPairs):
        srcDict = dicts['vocabs'][srcLangs[i]]
        tgtDict = dicts['vocabs'][tgtLangs[i]]
        
        #~ setID = uniqSrcLangs.index(srcLangs[i])
        
        srcID = int(dicts['srcLangs'].index(srcLangs[i]))
        tgtID = int(dicts['tgtLangs'].index(tgtLangs[i]))
        
        print srcID, tgtID
        
        setID = -1
        for i, cur_pair in enumerate(dicts['setIDs']):
            print cur_pair
            print cur_pair[0], cur_pair[1]
            if cur_pair[0] == srcID and cur_pair[1] == tgtID:
                setID = i
            
        assert setID >= 0, "Cannot find the language pair"
            #~ 
        #~ setID = dicts['setIDs'].index([srcID, tgtID])
        
        print('Preparing training ... for set %d ' % setID)
        
        srcSet, tgtSet = makeData(srcFiles[i], tgtFiles[i], 
                                  srcDict, tgtDict)
        
        train['src'].append(srcSet)
        train['tgt'].append(tgtSet)
            
        print('Preparing validation ... for set %d ' % i)
            
        validSrcSet, validTgtSet = makeData(validSrcFiles[i], validTgtFiles[i],
                                             srcDict, tgtDict)
        valid['src'].append(validSrcSet)
        valid['tgt'].append(validTgtSet)
        
        
        #~ 
        #~ # Sanity checks
        #~ assert len(srcLangs) == len(tgtLangs)
        #~ assert len(srcLangs) == len(srcFiles)
        #~ assert len(srcFiles) == len(tgtFiles)
        #~ 
        #~ langs = []
        #~ 
        #~ for lang in srcLangs + tgtLangs:
            #~ if not lang in langs:
                #~ langs.append(lang)
        #~ 
        #~ dicts['langs'] = langs
        #~ dicts['vocabs'] = dict()
        #~ dicts['nSets'] = len(srcLangs)
        #~ 
        #~ uniqSrcLangs = list(OrderedDict.fromkeys(srcLangs))
        #~ uniqTgtLangs = list(OrderedDict.fromkeys(tgtLangs))
        #~ 
        #~ dicts['srcLangs'] = uniqSrcLangs
        #~ dicts['tgtLangs'] = uniqTgtLangs
        #~ 
        #~ for lang in langs:
            #~ if lang not in dicts['vocabs']:
                #~ dataFilesWithLang = []
                #~ for i in range(len(srcFiles)):
                    #~ if srcLangs[i] == lang:
                        #~ dataFilesWithLang.append(srcFiles[i])
                    #~ if tgtLangs[i] == lang:
                        #~ dataFilesWithLang.append(tgtFiles[i])
                        #~ 
                #~ # We need to remove duplicate of this list 
                #~ 
                #~ sortedDataFiles = list(OrderedDict.fromkeys(dataFilesWithLang))
                #~ dicts['vocabs'][lang] = initVocabulary(lang, sortedDataFiles, 
                                                       #~ opt.vocab, opt.vocab_size)
        #~ 
        #~ # store the actual dictionaries for each side
        #~ dicts['src'] = dict()
        #~ dicts['tgt'] = dict()
#~ 
        #~ train = {}
        #~ train['src'] = list()
        #~ train['tgt'] = list()
        #~ dicts['setIDs'] = list()
        #~ dicts['setLangs'] = list()
        #~ 
        #~ valid = {}
        #~ valid['src'] = list()
        #~ valid['tgt'] = list()
    #~ 
        #~ for i in range(dicts['nSets']):
            #~ 
            #~ dicts['setIDs'].append([uniqSrcLangs.index(srcLangs[i]), uniqTgtLangs.index(tgtLangs[i])])
            #~ dicts['setLangs'].append([srcLangs[i], tgtLangs[i]])
            #~ 
            #~ srcID = dicts['setIDs'][i][0]
            #~ tgtID = dicts['setIDs'][i][1]
            #~ 
            #~ if srcID not in dicts['src']:
                #~ dicts['src'][srcID] = dicts['vocabs'][srcLangs[i]]
            #~ if tgtID not in dicts['tgt']:
                #~ dicts['tgt'][tgtID] = dicts['vocabs'][tgtLangs[i]]
            #~ 
            #~ srcDict = dicts['vocabs'][srcLangs[i]]
            #~ tgtDict = dicts['vocabs'][tgtLangs[i]]
            #~ 
            #~ print('Preparing training ... for set %d ' % i)
            #~ srcSet, tgtSet = makeData(srcFiles[i], tgtFiles[i], 
                                                                                                     #~ srcDict, tgtDict)
            #~ train['src'].append(srcSet)
            #~ train['tgt'].append(tgtSet)
            #~ 
            #~ print('Preparing validation ... for set %d ' % i)
            #~ 
            #~ validSrcSet, validTgtSet = makeData(validSrcFiles[i], validTgtFiles[i],
                                                                                                     #~ srcDict, tgtDict)
                                                                                                     #~ 
            #~ valid['src'].append(validSrcSet)
            #~ valid['tgt'].append(validTgtSet)
            #~ 
            #~ 
        #~ if opt.vocab is None:
            #~ print('Saving vocabularies ... ')
            #~ for lang in langs:
                #~ saveVocabulary(lang, dicts['vocabs'][lang], opt.save_data + '.dict.' + lang)
            #~ print('Done')
        #~ 
        #~ print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'type':  opt.src_type,
                 'train': train,
                 'valid': valid}
        
    torch.save(save_data, opt.save_data + '.train.pt')
    print('Finished.')
        

if __name__ == "__main__":
    main()
