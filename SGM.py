'''
This is an adaptation of SGM.py for Torch Lightning.

SGMnet is asimplified version of the SGM implemented so far.
Instead of creating two classes of networks (netA and netB, corresponding to encoder and decoder, or update and querry),
the present SGMnet class combines the two in a unified framework. Need to see wether it works, if it makes the lighting
pipeline smoother and whether it allows for Gestalt representation probing in a easy way.

[Created on 06.12.2021 by A. Lopopolo]
'''




import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from statistics import mean
from torch.autograd import Variable
from dataset import SGMDataset
from torch.utils.data import DataLoader, random_split
torch.autograd.set_detect_anomaly(True)




class SGMnet(pl.LightningModule):
    
    def __init__(self,
                 emb_size, 
                 hidden_size, 
                 output_size, 
                 probe_size, 
                 batch_size, 
                 nr_layers, 
                 dropout, 
                 learning_rate,
                 data,
                 ignore,
                 embeddings,
                 min_nr_words,
                 max_nr_words,
                 nr_probes,
                 nr_frames,
                 start_file,
                 max_nr_files,
                 nr_workers,
                 max_epochs,
                 optimizer,
                 voc_size  = 30002,
                 load_data = True):
        
        super().__init__()

        # parameters 
        self.optimizer      = optimizer
        
        self.emb_size      = emb_size
        
        self.nr_layers     = nr_layers
        
        self.hidden_size   = hidden_size
        
        self.output_size   = output_size
        
        self.probe_size    = probe_size
        
        self.batch_size    = batch_size
        
        self.learning_rate = learning_rate
        
        self.dropout       = dropout
        
        self.max_epochs    = max_epochs
        
        # data parameters
        self.data          = data
        
        self.ignore        = ignore
        
        self.embeddings    = embeddings
        
        self.min_words     = min_nr_words
        
        self.max_words     = max_nr_words
        
        self.nr_probes     = nr_probes
        
        self.nr_frames     = nr_frames
        
        self.start_file    = start_file
        
        self.max_nr_files  = max_nr_files
        
        self.nr_workers    = nr_workers
        
        self.nr_tags       = 26
        
        self.voc_size      = voc_size
        
        # load data
        if load_data:
        
            self.load_data()
        
            self.voc_size      = self.voc_size + 2
        
        # layers
        self.input         = nn.Embedding(num_embeddings=self.voc_size, embedding_dim=self.emb_size)
            
        if self.nr_layers > 1:
            
            self.gestalt       = nn.LSTM(self.emb_size, self.hidden_size, self.nr_layers, dropout=self.dropout, batch_first=True)
            
        else:
            
            self.gestalt       = nn.LSTM(self.emb_size, self.hidden_size, self.nr_layers, dropout=0.0, batch_first=True)
        
        self.dp_layer      = nn.Dropout(p=self.dropout, inplace=False)
        
        self.probe         = nn.Linear(in_features=self.probe_size, out_features=self.hidden_size)
        
        self.hidden        = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        
        self.rolefiller    = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
        
        # save hyperparameters
        self.save_hyperparameters()
         
      
    def forward(self, batch):
        
        previous_sg             = self.reset()
        
        x_in                    = self.input(batch[0])
            
        sg, previous_sg         = self.gestalt(x_in, previous_sg)

        sg                      = sg.view((sg.shape[0], 
                                           sg.shape[1], 
                                           1, 
                                           sg.shape[2])
                                         ).repeat(1, 1, batch[1].shape[2], 1)

        x_pr                    = torch.tanh(self.probe(batch[1]))

        x_pr_sg                 = x_pr + sg

        x_h                     = torch.tanh(self.hidden(x_pr_sg))

        rf_batch_hat            = torch.sigmoid(self.rolefiller(x_h))
        
        return rf_batch_hat
    
                
    def training_step(self, batch, batch_idx):

        previous_sg             = self.reset()
        
        x_in                    = self.input(batch[0])
            
        sg, previous_sg         = self.gestalt(x_in, previous_sg)
            
        sg                      = self.dp_layer(sg)

        sg                      = sg.view((sg.shape[0], 
                                           sg.shape[1], 
                                           1, 
                                           sg.shape[2])
                                         ).repeat(1, 1, batch[1].shape[2], 1)

        x_pr                    = torch.tanh(self.probe(batch[1]))

        x_pr_sg                 = x_pr + sg

        x_h                     = torch.tanh(self.hidden(x_pr_sg))

        rf_batch_hat            = torch.sigmoid(self.rolefiller(x_h))

        train_loss              = self.cross_entropy_loss(rf_batch_hat, batch[2])#self.loss(rf_batch_hat, batch[2])#
        
        self.log("train_loss", train_loss.data, sync_dist=True, on_epoch=True)
        
        return train_loss
     
        
    def validation_step(self, batch, batch_idx):

        previous_sg             = self.reset()

        x_in                    = self.input(batch[0])
              
        sg, previous_sg         = self.gestalt(x_in, previous_sg)

        sg                      = sg.view((sg.shape[0], 
                                           sg.shape[1], 
                                           1, 
                                           sg.shape[2])
                                         ).repeat(1, 1, batch[1].shape[2], 1)

        x_pr                    = torch.tanh(self.probe(batch[1]))

        x_pr_sg                 = x_pr + sg

        x_h                     = torch.tanh(self.hidden(x_pr_sg))

        rf_batch_hat            = torch.sigmoid(self.rolefiller(x_h))

        val_loss                = self.cross_entropy_loss(rf_batch_hat, batch[2])#self.loss(rf_batch_hat, batch[2])#
        
        self.log("val_loss", val_loss.data, sync_dist=True, on_epoch=True)
        
        return val_loss

              
    def configure_optimizers(self):
        
        if self.optimizer == 'Adamax':
            
            optimizer = optim.Adamax(self.parameters(), lr=self.learning_rate)
            
        elif self.optimizer == 'SGD':
            
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        
        lr_scheduler = {'scheduler': optim.lr_scheduler.OneCycleLR(optimizer,
                                                                   max_lr           = self.learning_rate,
                                                                   steps_per_epoch  = int(len(self.train_dataloader())),
                                                                   epochs           = self.max_epochs,
                                                                   anneal_strategy  = "linear",
                                                                   final_div_factor = 30,
                                                                   ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': int(len(self.train_dataloader())/3)}
        
        return [optimizer], [lr_scheduler]
              
              
    def encode(self, batch):
                
        previous_sg             = self.reset()
        
        x_in                    = self.input(batch[0])
              
        sg, previous_sg         = self.gestalt(x_in, previous_sg)

        return sg, previous_sg, x_in

    
    def reset(self):
        
        weight = next(self.parameters()).data
        
        return (Variable(weight.new(self.nr_layers, self.batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.nr_layers, self.batch_size, self.hidden_size).zero_()))
        
        '''
        self.previous_sg = (torch.zeros((self.nr_layers, self.batch_size, self.hidden_size),
                                        dtype=torch.float, device=self.device).detach(),
                            torch.zeros((self.nr_layers, self.batch_size, self.hidden_size), 
                                        dtype=torch.float, device=self.device).detach()
                           )
        '''
    
    
    def cross_entropy_loss(self, y_hat, y, epsilon=1e-12):
        """
        For non-mutually exclusive targets and a logsig output, the corresponding form for crossentropy is
        Xent2 = -sum( t.*log(y)) + (1-t).*log(1-y))
        """
    
        predictions = torch.clamp(y_hat, epsilon, 1.-epsilon)

        return -torch.mean(y * torch.log(y_hat+1e-9) + (1 - y) * torch.log(1 - y_hat+1e-9))
    
    
    def load_data(self):
        
        data = SGMDataset(data_path          = self.data, 
                          ignore_filelist    = self.ignore, 
                          embedding_filename = self.embeddings, 
                          min_nr_words       = self.min_words,
                          max_nr_words       = self.max_words, 
                          nr_probes          = self.nr_probes, 
                          nr_frames          = self.nr_frames,
                          nr_tags            = self.nr_tags,
                          batch_size         = self.batch_size,
                          start_file         = self.start_file,
                          max_nr_files       = self.max_nr_files,
                          empty              = False
                          )
        
        self.voc_size = data.voc_size
        
        train_set_size = int(len(data) * 0.9)
        
        valid_set_size = len(data) - train_set_size
        
        self.train_data, self.valid_data = random_split(data, 
                                                        [train_set_size, valid_set_size],
                                                        generator=torch.Generator().manual_seed(42)
                                                       )
    
    
    def train_dataloader(self):

        loader = DataLoader(self.train_data, 
                            batch_size  = self.batch_size, 
                            shuffle     = True, 
                            num_workers = self.nr_workers, 
                            pin_memory  = True,
                            drop_last   = True
                           )
        
        return loader

    
    def val_dataloader(self):
        
        loader = DataLoader(self.valid_data, 
                            batch_size  = self.batch_size, 
                            shuffle     = False, 
                            num_workers = self.nr_workers, 
                            pin_memory  = True,
                            drop_last   = True
                           )
        
        return loader
    
    
    