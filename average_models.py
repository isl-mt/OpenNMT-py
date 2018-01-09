from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-models', required=True,
                    help='Path to model .pt file')
parser.add_argument('-output', default='model.averaged',
                    help="""Path to output averaged model""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
                    
def build_model(model_opt, model_state, generator_state, dicts, nSets, cuda):
    encoder = onmt.Models.Encoder(model_opt, dicts['src'])
    decoder = onmt.Models.Decoder(model_opt, dicts['tgt'], nSets)
    this_model = onmt.Models.NMTModel(encoder, decoder)
    generator = onmt.Models.Generator(model_opt, dicts['tgt'])
    
    # load weights
    this_model.load_state_dict(model_state)
    generator.load_state_dict(generator_state)
    
    # ship to cuda if necessary
    if cuda:
        this_model.cuda()
        generator.cuda()
    else:
        this_model.cpu()
        generator.cpu()

    this_model.generator = generator
    
    return this_model
                    
def main():
    
    opt = parser.parse_args()
    
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    
    # opt.model should be a string of models, split by |
        
    models = opt.models.split("|")
    #~ print(models)
    n_models = len(models)
    
    print("Loading main model from %s ..." % models[0])
    checkpoint = torch.load(models[0], map_location=lambda storage, loc: storage)
    
    if 'optim' in checkpoint:
        del checkpoint['optim']
    
    main_checkpoint = checkpoint
    model_opt = checkpoint['opt']
    dicts = checkpoint['dicts']
    nSets = dicts['nSets']
    
    main_model = build_model(model_opt, main_checkpoint['model'], main_checkpoint['generator'], dicts, nSets, opt.cuda)
    
    for i in range(1, len(models)):
        
        model = models[i]
        print("Loading model from %s ..." % models[i])
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage    )

        model_opt = checkpoint['opt']

        # delete optim information to save GPU memory
        if 'optim' in checkpoint:
            del checkpoint['optim']
        
        current_model = build_model(model_opt, checkpoint['model'], checkpoint['generator'], dicts, nSets, opt.cuda)
        
        # Sum the parameter values 
        for (main_param, param) in zip(main_model.parameters(), current_model.parameters()):
            main_param.data.add_(param.data)
    
    # Normalizing
    for main_param in main_model.parameters():
        main_param.data.div_(n_models) 
    
    # Saving
    model_state_dict = main_model.state_dict()
    
    model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
    generator_state_dict = main_model.generator.state_dict()
                            
    
    save_checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dicts,
            'opt': model_opt,
            'epoch': -1,
            'iteration' : -1,
            'batchOrder' : None
    }
    
    torch.save(save_checkpoint, opt.output)
    
                    

    
    
if __name__ == "__main__":
    main()
    
    
                                     
