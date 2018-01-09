import onmt
import onmt.modules
import torch.nn as nn
import torch
from torch.autograd import Variable

# Ensemble decoding


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


class Rescorer(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self._type = "text"
        
        # opt.model should be a string of models, split by |
        
        models = opt.model.split("|")
        print(models)
        self.n_models = len(models)
        
        # only one src and target language
        
        self.models = list()
        self.logSoftMax = torch.nn.LogSoftmax()
        nSets = 0
        
        for i, model in enumerate(models):
            checkpoint = torch.load(model)

            model_opt = checkpoint['opt']
            
            del checkpoint['optim']
            
            # assuming that all these models use the same dict
            # the first checkpoint's dict will be loaded
            if i == 0:
                self.dicts = checkpoint['dicts']
                self.src_dict = self.dicts['vocabs'][opt.src_lang]
                self.tgt_dict = self.dicts['vocabs'][opt.tgt_lang]
                nSets = self.dicts['nSets']
            
            
            # Build the model
            encoder = onmt.Models.Encoder(model_opt, self.dicts['src'])
            decoder = onmt.Models.Decoder(model_opt, self.dicts['tgt'], nSets)
            this_model = onmt.Models.NMTModel(encoder, decoder)

            generator = onmt.Models.Generator(model_opt, self.dicts['tgt'])

            this_model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])

            if opt.cuda:
                this_model.cuda()
                generator.cuda()
            else:
                this_model.cpu()
                generator.cpu()

            this_model.generator = generator

            this_model.eval()
            
            # Need to find the src and tgt id
            srcID = self.dicts['srcLangs'].index(opt.src_lang)
            tgtID = self.dicts['tgtLangs'].index(opt.tgt_lang)
            
            # After that, look for the pairID
            
            setIDs = self.dicts['setIDs']
            
            pair = -1
            for i, sid in enumerate(setIDs):
                if sid[0] == srcID and sid[1] == tgtID:
                    pair = i
                    break
                            
            assert pair >= 0, "Cannot find any language pair with your provided src and tgt id"
            print(" * Translating with pair %i " % pair)
            #~ print(srcID, tgtID)
            #~ print(self.model)
            this_model.switchLangID(srcID, tgtID)
            this_model.switchPairID(pair) 
            
            self.models.append(this_model)


    def _getBatchSize(self, batch):
        if self._type == "text":
            return batch.size(1)
        else:
            return batch.size(0)
    

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.
        if self._type == "text":
            srcData = [self.src_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD)
                       for b in srcBatch]
        elif self._type == "img":
            srcData = [transforms.ToTensor()(
                Image.open(self.opt.src_img_dir + "/" + b[0]))
                       for b in srcBatch]

        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData, self.opt.batch_size,
                            self.opt.cuda, volatile=True,
                            data_type=self._type, balance=False)

    def rescoreBatch(self, srcBatch, tgtBatch):
        # Batch size is in different location depending on data.
        
        contexts = dict()
        encStates = dict()


        #  (1) run the encoders on the src
        for i in xrange(self.n_models):
            states, contexts[i] = self.models[i].encoder(srcBatch)
            
            # reshape the states
            encStates[i] = (self.models[i]._fix_enc_hidden(states[0]),
                     self.models[i]._fix_enc_hidden(states[1]))

        # Drop the lengths needed for encoder.
        #~ print(srcBatch[1])
        srcBatch = srcBatch[0]
        
        batchSize = self._getBatchSize(srcBatch)

        rnnSizes = dict()
        for i in xrange(self.n_models):
            rnnSizes[i] = contexts[i].size(2)
        
        #~ decoder = self.model.decoder
        #~ attentionLayer = decoder.attn.current()
        useMasking = ( self._type == "text" and batchSize > 1 )

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = None
        if useMasking:
            padMask = srcBatch.data.eq(onmt.Constants.PAD).t()

        def mask(padMask):
            if useMasking:
                for i in xrange(self.n_models):
                    self.models[i].decoder.attn.current().applyMask(padMask)
                #~ attentionLayer.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = contexts[0].data.new(batchSize).zero_()
        
        # currently gold scoring is only supported when using single model decoding
        if tgtBatch is not None and self.n_models == 1:
            decStates = encStates[0]
            this_model = self.models[0]
            context = contexts[0]

            decOut = this_model.make_init_decoder_output(context)
            mask(padMask)
            initOutput = this_model.make_init_decoder_output(context)
            decOut, decStates, attn = this_model.decoder(
                tgtBatch[:-1], decStates, context, initOutput)
            for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
                gen_t = this_model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

        

        return goldScores

    def rescore(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        
        allScores = list()
        
        for i in xrange(dataset.numBatches):
            src, tgt, indices = dataset[i]
            batchSize = self._getBatchSize(src[0])

            #  (2) rescore
            goldScore = self.rescoreBatch(src, tgt)
        
            goldScore = list(list(zip(
                *sorted(zip(goldScore, indices),
                        key=lambda x: x[-1])))[:-1][0])
        
            allScores += goldScore
        #~ print(allScores)
        

        return allScores
