'''
Training script for SGM with Torch Lightning.

[Created on 06.12.2021 by A. Lopopolo]
'''



from SGM import SGMnet
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import json
import warnings
import argparse
import torch

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")



def list_full_paths(directory):
    
    return [os.path.join(directory, file) for file in os.listdir(directory)]



def main(args):

    # create the directory (if not already there)
    if not os.path.exists(args.log_directory):
        
        os.makedirs(args.log_directory)

        
    # initialize the model
    rolefiller_size = 300 + args.nr_tags + args.nr_frames
    
    
    # continue training the model on the same data
    if args.continue_training == 'True':
        
        # find the last trained ckpt
        ckpts                 = list_full_paths("%strained_models/" % args.log_directory)
        ckpts.sort(key=lambda x: os.path.getmtime(x))
        args.last_ckpt_sgm    = ckpts[-1]
        
        if '_ptLM' in args.lst_ckpt_sgm:
            
            args.id = args.id + '_ptLM'
        
        # new number of epochs
        args.e                = args.e + torch.load(args.last_ckpt_sgm, map_location='cpu')['epoch']
        
        # determine training files
        args.start_file       = torch.load(args.last_ckpt_sgm, map_location='cpu')['hyper_parameters']['start_file'] 
        
        # instantiate the SGM
        SGM = SGMnet(emb_size      = args.i,
                     hidden_size   = args.g, 
                     output_size   = rolefiller_size,  
                     probe_size    = rolefiller_size, 
                     batch_size    = args.batch_size,
                     nr_layers     = args.l,
                     dropout       = args.dp,
                     learning_rate = args.lr,
                     embeddings    = args.embeddings,
                     data          = args.data,
                     ignore        = args.ignore,
                     min_nr_words  = args.min_words,
                     max_nr_words  = args.max_words,
                     nr_probes     = args.nr_probes,
                     nr_frames     = args.nr_frames,
                     start_file    = args.start_file,
                     max_nr_files  = args.max_nr_files,
                     nr_workers    = args.nr_workers,
                     max_epochs    = args.e,
                     optimizer     = args.optimizer,
                     load_data     = True
                     )
        
    
        # check points
        checkpoint_callback = ModelCheckpoint(filename  = "SGM=%s" % args.id + "-{epoch:03d}-{val_loss:.4f}-{train_loss:.4f}",
                                              dirpath  = "%strained_models/" % args.log_directory,
                                              every_n_epochs  = 1,
                                              save_top_k  = -1
                                              )

        # train the model
        trainer = pl.Trainer(accelerator             = 'gpu',
                             auto_select_gpus        = True,
                             gpus                    = 4,
                             max_epochs              = args.e,
                             check_val_every_n_epoch = 1,
                             gradient_clip_val       = 0.25,
                             strategy                = DDPStrategy(find_unused_parameters=False),
                             callbacks               = [checkpoint_callback],
                             default_root_dir        = args.log_directory
                             )
    
        trainer.fit(SGM, ckpt_path=args.last_ckpt_sgm)
        
        
    # continue training the model on a new batch of data ('fine tune')
    if args.fine_tune == 'True':
        
        # find the last trained ckpt
        ckpts                = list_full_paths("%strained_models/" % args.log_directory)
        ckpts.sort(key=lambda x: os.path.getmtime(x))
        args.last_ckpt_sgm   = ckpts[-1]
        
        # load the pretrained SGM
        ptSGM                = torch.load(args.last_ckpt_sgm,map_location='cpu')
        
        if '_ptLM' in args.last_ckpt_sgm:
            
            args.id = args.id + '_ptLM'
        
        # determine fine_tuning training files
        model_start_file     = ptSGM['hyper_parameters']['start_file']
        model_max_nr_files   = ptSGM['hyper_parameters']['max_nr_files']
        
        args.start_file      = model_start_file + model_max_nr_files 
        
        # instantiate the SGM
        SGM = SGMnet(emb_size      = args.i,
                     hidden_size   = args.g, 
                     output_size   = rolefiller_size,  
                     probe_size    = rolefiller_size, 
                     batch_size    = args.batch_size,
                     nr_layers     = args.l,
                     dropout       = args.dp,
                     learning_rate = args.lr,
                     embeddings    = args.embeddings,
                     data          = args.data,
                     ignore        = args.ignore,
                     min_nr_words  = args.min_words,
                     max_nr_words  = args.max_words,
                     nr_probes     = args.nr_probes,
                     nr_frames     = args.nr_frames,
                     start_file    = args.start_file,
                     max_nr_files  = args.max_nr_files,
                     nr_workers    = args.nr_workers,
                     max_epochs    = args.e,
                     optimizer     = args.optimizer,
                     load_data     = True
                     )
        
        # assigne weights
        SGM.input.weight         = torch.nn.Parameter(ptSGM['state_dict']['input.weight'])
        SGM.gestalt.weight_ih_l0 = torch.nn.Parameter(ptSGM['state_dict']['gestalt.weight_ih_l0'])
        SGM.gestalt.weight_hh_l0 = torch.nn.Parameter(ptSGM['state_dict']['gestalt.weight_hh_l0'])
        SGM.gestalt.bias_ih_l0   = torch.nn.Parameter(ptSGM['state_dict']['gestalt.bias_ih_l0'])
        SGM.gestalt.bias_hh_l0   = torch.nn.Parameter(ptSGM['state_dict']['gestalt.bias_hh_l0'])
        SGM.probe.weight         = torch.nn.Parameter(ptSGM['state_dict']['probe.weight'])
        SGM.hidden.weight        = torch.nn.Parameter(ptSGM['state_dict']['hidden.weight'])
        SGM.rolefiller.weight    = torch.nn.Parameter(ptSGM['state_dict']['rolefiller.weight'])
        
        # check points
        checkpoint_callback = ModelCheckpoint(filename  = "SGM=%s" % args.id + "-{epoch:03d}-{val_loss:.4f}-{train_loss:.4f}",
                                              dirpath   = "%strained_models/" % args.log_directory,
                                              every_n_epochs  = 1,
                                              save_top_k  = -1
                                              )

        # train the model
        trainer = pl.Trainer(accelerator             = 'gpu',
                             auto_select_gpus        = True,
                             gpus                    = 4,
                             max_epochs              = args.e,
                             check_val_every_n_epoch = 1,
                             gradient_clip_val       = 0.25,
                             strategy                = DDPStrategy(find_unused_parameters=False),
                             callbacks               = [checkpoint_callback],
                             default_root_dir        = args.log_directory
                             )
    
        trainer.fit(SGM)
                
        
    if args.pretrained_lm is not None:
   
        print('\nUsing pretrained LM:\n%s\n' % args.pretrained_lm)
    
        args.id = args.id + '_ptLM'
        
        # load the pretrained LM
        ptLM = torch.load(args.pretrained_lm)
        
        args.start_file = ptLM['hyper_parameters']['start_file'] + ptLM['hyper_parameters']['max_nr_files']
        
        # instantiate the SGM
        SGM = SGMnet(emb_size      = args.i,
                     hidden_size   = args.g, 
                     output_size   = rolefiller_size,  
                     probe_size    = rolefiller_size, 
                     batch_size    = args.batch_size,
                     nr_layers     = args.l,
                     dropout       = args.dp,
                     learning_rate = args.lr,
                     embeddings    = args.embeddings,
                     data          = args.data,
                     ignore        = args.ignore,
                     min_nr_words  = args.min_words,
                     max_nr_words  = args.max_words,
                     nr_probes     = args.nr_probes,
                     nr_frames     = args.nr_frames,
                     start_file    = args.start_file,
                     max_nr_files  = args.max_nr_files,
                     nr_workers    = args.nr_workers,
                     max_epochs    = args.e,
                     optimizer     = args.optimizer
                     )
        
        # assigne weights
        SGM.input.weight         = torch.nn.Parameter(ptLM['state_dict']['input.weight'])
        SGM.gestalt.weight_ih_l0 = torch.nn.Parameter(ptLM['state_dict']['gestalt.weight_ih_l0'])
        SGM.gestalt.weight_hh_l0 = torch.nn.Parameter(ptLM['state_dict']['gestalt.weight_hh_l0'])
        SGM.gestalt.bias_ih_l0   = torch.nn.Parameter(ptLM['state_dict']['gestalt.bias_ih_l0'])
        SGM.gestalt.bias_hh_l0   = torch.nn.Parameter(ptLM['state_dict']['gestalt.bias_hh_l0'])        


        # check points
        checkpoint_callback = ModelCheckpoint(filename  = "SGM=%s" % args.id + "-{epoch:03d}-{val_loss:.4f}-{train_loss:.4f}",
                                              dirpath   = "%strained_models/" % args.log_directory,
                                              every_n_epochs  = 1,
                                              save_top_k  = -1
                                             )
    

        # train the model
        trainer = pl.Trainer(accelerator             = 'gpu',
                             auto_select_gpus        = True,
                             gpus                    = 4,
                             max_epochs              = args.e,
                             check_val_every_n_epoch = 1,
                             gradient_clip_val       = 0.25,
                             strategy                = DDPStrategy(find_unused_parameters=False),
                             callbacks               = [checkpoint_callback],
                             default_root_dir        = args.log_directory
                             )

        trainer.fit(SGM)
    
    
    if args.fine_tune == 'False' and args.continue_training == 'False' and args.pretrained_lm is None:
        
        # instantiate the SGM
        SGM = SGMnet(emb_size      = args.i,
                     hidden_size   = args.g, 
                     output_size   = rolefiller_size,  
                     probe_size    = rolefiller_size, 
                     batch_size    = args.batch_size,
                     nr_layers     = args.l,
                     dropout       = args.dp,
                     learning_rate = args.lr,
                     embeddings    = args.embeddings,
                     data          = args.data,
                     ignore        = args.ignore,
                     min_nr_words  = args.min_words,
                     max_nr_words  = args.max_words,
                     nr_probes     = args.nr_probes,
                     nr_frames     = args.nr_frames,
                     start_file    = args.start_file,
                     max_nr_files  = args.max_nr_files,
                     nr_workers    = args.nr_workers,
                     max_epochs    = args.e,
                     optimizer     = args.optimizer,
                     load_data     = True
                     )
        
        # check points
        checkpoint_callback = ModelCheckpoint(filename  = "SGM=%s" % args.id + "-{epoch:03d}-{val_loss:.4f}-{train_loss:.4f}",
                                              dirpath   = "%strained_models/" % args.log_directory,
                                              every_n_epochs  = 1,
                                              save_top_k  = -1
                                              )

        # train the model
        trainer = pl.Trainer(accelerator             = 'gpu',
                             auto_select_gpus        = True,
                             gpus                    = 4,
                             max_epochs              = args.e,
                             check_val_every_n_epoch = 1,
                             gradient_clip_val       = 0.25,
                             strategy                = DDPStrategy(find_unused_parameters=False),
                             callbacks               = [checkpoint_callback],
                             default_root_dir        = args.log_directory
                             )
    
        trainer.fit(SGM)

    
    with open(args.log_directory + 'SGM=%s_arguments.txt' % args.id, 'w') as argfile:
        
        json.dump(args.__dict__, argfile, indent=2)
        
    argfile.close()


    
    
if __name__ == '__main__':
    
    """
    TODO: 1) dutch fasttext, float and binarized 
    2) check if padding and ignore zeros is implemented in the training here and if it works
    """

    parser = argparse.ArgumentParser(description='pytorch: recurrent nerual network language model')
    
    # model ID
    parser.add_argument('--id',
                        type=str, default='S0',
                        help='model id')
    
    # Optimizer
    parser.add_argument('--optimizer',
                        type=str, default='Adamax',
                        help='optimizer')
    
    # Embeddings ***** (need to be created for NL)
    parser.add_argument('--embeddings', 
                        type=str, default='../FEATURES/fasttext-wiki-news-300d-30k-thr.txt',
                        help='embedding model filename')
    
    # Vocabulary
    parser.add_argument('--vocabulary', 
                        type=str, default='../LEXICA/lexicon-wordforms-RWEN2-TOTAL.top30K.min4-max25-words.max5-frames.fasstext-wiki-news-300d-1M-thr-lexicon.xlsx',
                        help='vocabulary filename')
    
    # Data path
    parser.add_argument('--data', 
                        type=str, default='../../../data/rw-eng-2-total-sgm-argheads/',
                        help='path to the training data')
    
    # List of data files to ignore *****
    parser.add_argument('--ignore', 
                        type=str, default='../EXPERIMENTS/MEG/stimulus_filelist.txt',
                        help='file containing a list of files to be ignored for training')
    
    # LOGS
    parser.add_argument('--log_directory', 
                        type=str, default='../TRAINED_MODELS/SGM/MODELS_%iFs/',
                        help='path to the log and trained models')
    
    # pretrained LM *****
    parser.add_argument('--pretrained_lm', 
                        type=str, default=None,
                        help='path to a pretrained lm')
    
    parser.add_argument('--continue_training', 
                        type=str, default='False',
                        help='continue training (on the same data) from the last epoch')
    
    parser.add_argument('--fine_tune', 
                        type=str, default='False',
                        help='continue training (on NEW data) from the last epoch')
    
    parser.add_argument('-i', 
                        type=int, default=300,
                        help='the size of the input layer')
    
    parser.add_argument('-g',
                        type=int, default=600,
                        help='the size of the hidden layer')
    
    parser.add_argument('-l', 
                        type=int, default=1,
                        help='the number of the hidden layer')
    
    parser.add_argument('--lr', 
                        type=float, default=0.0005,
                        help='learning rate')
    
    parser.add_argument('--dp', 
                        type=float, default=0.0,
                        help='drop out')
    
    parser.add_argument('-e', 
                        type=int, default=100,
                        help='number of training epochs')
    
    parser.add_argument('-batch_size', 
                        type=int, default=32,
                        help='the size of the batch')
    
    # Max Number of frames (or "events", or "predicates") in a sentence
    # Try setting it to the actual max nr of frames in your NL data (6 or 8????)
    parser.add_argument('--nr_frames', 
                        type=int, default=10,
                        help='the max nr of frames per sentence')
    
    # Number of Roles (including PRD) in the dataset, i.e. 33 in NL dataset
    parser.add_argument('--nr_tags', 
                        type=int, default=26,
                        help='nr of tags')
    
    # I think you can simply set it to 0 ****
    parser.add_argument("--min_words", 
                        type=int, default=4,
                        help='min nr of words per sentence')
    
    # ... (maybe set it to the length of the longest sentence in the NL dataset?) ****
    parser.add_argument("--max_words", 
                        type=int, default=20,
                        help='max nr of words per sentence')
    
    # ... (maybe set it to the max nr of probes in the NL dataset?) ****
    parser.add_argument("--nr_probes", 
                        type=int, default=10,
                        help='max nr of probes')
    
    # check this (in theory is should not matter, as far as the value is > 0)
    parser.add_argument('--max_nr_files', 
                        type=int, default=3000,
                        help='the max number of files for training')
    
    parser.add_argument('--nr_workers', 
                        type=int, default=3,
                        help='number of workers for dataloader')
    
    
    args = parser.parse_args()
    args.log_directory = args.log_directory % args.max_nr_files
    
    
    print('\n', args)
    
   
    # run MAIN
    main(args)
    
    
    
    
