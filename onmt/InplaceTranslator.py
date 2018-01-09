import onmt
import onmt.modules
import torch.nn as nn
import torch
from torch.autograd import Variable


# A class for translating during validation
# Not for effectively loading model and translate
class InplaceTranslator(object):
    
    def __init__(self, model, dicts, beam_size=1, cuda=True):
        
        self.model = model
        self.dicts = dicts
        self.beam_size = beam_size
        self.cuda = cuda
        self.n_best = beam_size
        
        self.max_sent_length = 100
        
        self.tt = torch.cuda if self.cuda else torch
        
            
    def switchPair(sid, setIDs):
        self.model.switchLangID(setIDs[sid][0], setIDs[sid][1])
        self.model.switchPairID(sid)


    def _getBatchSize(self, batch):
        
        return batch.size(1)
    
    def translateBatch(self, srcBatch):
        # Batch size is in different location depending on data.
        beamSize = self.beam_size

        #  (1) run the encoders on the src
        
        states, context = self.model.encoder(srcBatch)

        # reshape the states
        encStates = (self.model._fix_enc_hidden(states[0]),
                 self.model._fix_enc_hidden(states[1]))

        # Drop the lengths needed for encoder.
        srcBatch = srcBatch[0]
        batchSize = self._getBatchSize(srcBatch)

        rnnSize = context.size(2)
        
        #~ decoder = self.model.decoder
        #~ attentionLayer = decoder.attn.current()
        useMasking = ( batchSize > 1 )

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = None
        if useMasking:
            padMask = srcBatch.data.eq(onmt.Constants.PAD).t()

        def mask(padMask):
            if useMasking:
                #~ attentionLayer.applyMask(padMask)
                self.model.decoder.attn.current().applyMask(padMask)

        #  (2) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        
        context = Variable(context.data.repeat(1, beamSize, 1))
        
        decStates = (Variable(encStates[0].data.repeat(1, beamSize, 1)),
                         Variable(encStates[1].data.repeat(1, beamSize, 1)))
                    

        
        # Initialize the beams
        # Each beam is an object containing the translation status for each sentence in the batch
        beam = [onmt.Beam(beamSize, self.cuda) for k in range(batchSize)]
        
        # Here we prepare the decoder output (zeroes)
        # For input feeding
        decOuts = self.model.make_init_decoder_output(context)
        

        if useMasking:
            padMask = srcBatch.data.eq(
                onmt.Constants.PAD).t() \
                                   .unsqueeze(0) \
                                   .repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.max_sent_length):
            mask(padMask)
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)
                                 #~ 
            #~ print(context.size())
            #~ 
            #~ print(decOuts.size())
                                 
            # compute new decoder output (distribution)
            decOuts, decStates, attn = self.model.decoder(
                    Variable(input, volatile=True), decStates, context, decOuts)
            
            # decOut: 1 x (beam*batch) x numWords
            decOuts = decOuts.squeeze(0)        
            out = self.model.generator.forward(decOuts)
           
            

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
                
                for decState in decStates:  # iterate over h, c
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
            
            decStates = (updateActive(decStates[0], rnnSize),
                         updateActive(decStates[1], rnnSize))
            decOuts = updateActive(decOuts, rnnSize)
            context = updateActive(context, rnnSize)
            if useMasking:
                padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.n_best

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
        
        if useMasking:
            self.model.decoder.attn.current().applyMask(None)

        return allHyp, allScores, allAttn

    def translate(self, srcBatch):
        #  (1) convert words to indexes
        src = srcBatch
        batchSize = self._getBatchSize(src[0])

        #  (2) translate
        pred, predScore, attn = self.translateBatch(src)

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            # only take the top of the beam search - for simplicity
            predBatch.append(pred[b][0])

        return predBatch
