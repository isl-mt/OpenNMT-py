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


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None
        self._type = "text"
        self.ensemble_op = opt.ensemble_op
        
        # opt.model should be a string of models, split by |
        
        models = opt.model.split("|")
        print(models)
        self.n_models = len(models)
        
        # only one src and target language
        
        self.models = list()
        self.logSoftMax = torch.nn.LogSoftmax()
        nSets = 0
        
        for i, model in enumerate(models):
            if opt.verbose:
                print('Loading model from %s' % opt.model)
            checkpoint = torch.load(opt.model,
                               map_location=lambda storage, loc: storage)
        
            if opt.verbose:
                print('Done')

            model_opt = checkpoint['opt']
            
            # delete optim information to save GPU memory
            if 'optim' in checkpoint:
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

            #~ self.model = model
            #~ self.model.eval()
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

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def _getBatchSize(self, batch):
        if self._type == "text":
            return batch.size(1)
        else:
            return batch.size(0)
    
    # Combine distributions from different models
    def _combineOutputs(self, outputs):
        
        if len(outputs) == 1:
            return outputs[0]
        
        if self.ensemble_op == "logSum":
            output = (outputs[0])
            
            # sum the log prob
            for i in range(1, len(outputs)):
                output += (outputs[i])
                
            output.div(len(outputs))
            
            #~ output = torch.log(output)
            output = self.logSoftMax(output)
        elif self.ensemble_op == "sum":
            output = torch.exp(outputs[0])
            
            # sum the log prob
            for i in range(1, len(outputs)):
                output += torch.exp(outputs[i])
                
            output.div(len(outputs))
            
            #~ output = torch.log(output)
            output = torch.log(output)
        else:
            raise ValueError('Emsemble operator needs to be "sum" or "logSum", the current value is %s' % self.ensemble_op)

        
        return output
    
    # Take the average of attention scores
    def _combineAttention(self, attns):
        
        attn = attns[0]
        
        for i in range(1, len(attns)):
            attn += attns[i]
        
        attn.div(len(attns))
        
        return attn

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

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        #~ tokens = tokens[:-1]  # EOS
        if tokens[-1] == onmt.Constants.EOS_WORD:
            tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):
        # Batch size is in different location depending on data.
        
        contexts = dict()
        encStates = dict()

        beamSize = self.opt.beam_size

        #  (1) run the encoders on the src
        for i in xrange(self.n_models):
            states, contexts[i] = self.models[i].encoder(srcBatch)
            
            # reshape the states
            encStates[i] = (self.models[i]._fix_enc_hidden(states[0]),
                     self.models[i]._fix_enc_hidden(states[1]))

        # Drop the lengths needed for encoder.
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

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        
        decStates = dict()
        
        for i in xrange(self.n_models):
            contexts[i] = Variable(contexts[i].data.repeat(1, beamSize, 1))
        
            decStates[i] = (Variable(encStates[i][0].data.repeat(1, beamSize, 1)),
                         Variable(encStates[i][1].data.repeat(1, beamSize, 1)))
        
        # Initialize the beams
        # Each beam is an object containing the translation status for each sentence in the batch
        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        
        # Here we prepare the decoder output (zeroes)
        # For input feeding
        decOuts = dict()
        attns = dict()
        outs = dict()
        for i in xrange(self.n_models):
            decOuts[i] = self.models[i].make_init_decoder_output(contexts[i])

        if useMasking:
            padMask = srcBatch.data.eq(
                onmt.Constants.PAD).t() \
                                   .unsqueeze(0) \
                                   .repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):
            mask(padMask)
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)
                                 
            # compute new decoder output (distribution)
            for i in xrange(self.n_models):
                decOuts[i], decStates[i], attns[i] = self.models[i].decoder(
                    Variable(input, volatile=True), decStates[i], contexts[i], decOuts[i])
                # decOut: 1 x (beam*batch) x numWords
                decOuts[i] = decOuts[i].squeeze(0)
                outs[i] = self.models[i].generator.forward(decOuts[i])
            
            # combine outputs and attention
            
            out = self._combineOutputs(outs)
            
            attn = self._combineAttention(attns)
            

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1) \
                        .transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1) \
                       .transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]
                
                for i in xrange(self.n_models):
                    for decState in decStates[i]:  # iterate over h, c
                        # layers x beam*sent x dim
                        sentStates = decState.view(-1, beamSize,
                                                   remainingSents,
                                                   decState.size(2))[:, :, idx]
                        sentStates.data.copy_(
                            sentStates.data.index_select(
                                1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t, size):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, size)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx)
                                .view(*newSize), volatile=True)
            
            for i in xrange(self.n_models):
                decStates[i] = (updateActive(decStates[i][0], rnnSizes[i]),
                             updateActive(decStates[i][1], rnnSizes[i]))
                decOuts[i] = updateActive(decOuts[i], rnnSizes[i])
                contexts[i] = updateActive(contexts[i], rnnSizes[i])
            if useMasking:
                padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            allHyp += [hyps]
            if useMasking:
                valid_attn = srcBatch.data[:, b].ne(onmt.Constants.PAD) \
                                                .nonzero().squeeze(1)
                attn = [a.index_select(1, valid_attn) for a in attn]
            allAttn += [attn]

            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                    ["%4f" % s for s in t.tolist()]
                    for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_dict.getLabel(id)
                      for id in t.tolist()]
                     for t in beam[b].nextYs][1:])
        
        mask(None)

        return allHyp, allScores, allAttn, goldScores

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        src, tgt, indices = dataset[0]
        batchSize = self._getBatchSize(src[0])

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(src, tgt)
        pred, predScore, attn, goldScore = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore
