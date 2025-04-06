        
from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np
import glob, os
from itertools import zip_longest
from gensim.models import KeyedVectors
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)

### Base Dataset ###
class Dataset(Dataset):
    
    def __init__(self):
        
        super(Dataset, self).__init__()
        
        self.inputs = None
        
        self.probes = None
        
        self.outputs = None
        
        
    def __len__(self):
        
        return len(self.samples)

    
    def __getitem__(self, idx):
        
        return self.inputs[idx][0], self.probes[idx][0], self.outputs[idx][0]
    
    
    def subset(self, start, stop):
        
        new_dataset = pdpDataset()
        
        new_dataset.inputs = self.inputs[start:stop]
        
        new_dataset.probes = self.probes[start:stop]
        
        new_dataset.outputs = self.outputs[start:stop]
        
        new_dataset.nr_words = new_dataset.inputs.shape[0]
        
        return new_dataset
    
    
    def loadVocabulary(self, lexicon_file):
        
        self.vocabulary = dict()
        
        with open(lexicon_file, 'r') as lex_in:
            
            for i,line in enumerate(lex_in.readlines()):
                
                self.vocabulary[line.split()[0]] = i+1
                
        self.voc_size = len(self.vocabulary)
    
    

    
### SGM DATASET ###
class SGMDataset(Dataset):
    
    def __init__(self, 
                 data_path, 
                 ignore_filelist, 
                 embedding_filename, 
                 min_nr_words, 
                 max_nr_words, 
                 nr_probes, 
                 nr_tags, 
                 nr_frames, 
                 batch_size, 
                 start_file, 
                 max_nr_files,
                 empty=False):
        
        super(SGMDataset, self).__init__()
        
        self.nr_tags            = nr_tags
        
        self.sentence_length    = range(min_nr_words,max_nr_words+1)
        
        self.nr_probes          = range(0,nr_probes+1)
        
        self.max_nr_words       = max_nr_words
        
        self.max_nr_frames      = nr_frames
        
        self.emb_model          = KeyedVectors.load_word2vec_format(embedding_filename)
        
        self.voc_size           = len(list(self.emb_model.index_to_key))
        
        self.inputs             = []
        
        self.targets            = []
        
        self.probes             = []  
        
        self.batch_size         = batch_size
        
        self.size               = None
        
        self.ignore_filelist    = ignore_filelist
        
        self.start_file         = start_file
        
        self.max_nr_files       = max_nr_files
        
        if not empty:
        
            if os.path.isfile(data_path):
            
                self.loadDataFile(data_path)
            
                print('\n')
            
            elif os.path.isdir(data_path):
            
                self.loadDataDir(data_path)
              
                print('\n')
        
            self.size = len(self.inputs)

            
    def __len__(self):
        
        return len(self.inputs)

    
    def __getitem__(self, idx):
                
        return self.encode_data(self.inputs[idx], self.probes[idx], self.targets[idx])
    
    
    def loadDataDir(self, dir_name):
        
        file_list = [f for f in glob.glob(dir_name + '*.inlr.txt', recursive=False)]
        file_list.sort()

        if self.ignore_filelist != None:
            
            ignore_filenames = open(self.ignore_filelist, 'r').readlines()
            ignore_filenames = [f.strip('\n') for f in ignore_filenames]
            file_list = [f for f in file_list if f not in ignore_filenames]
            
        if self.max_nr_files != None:
            
            file_list = file_list[self.start_file:self.start_file+self.max_nr_files]
        
        for f, file_name in enumerate(file_list):

            self.loadDataFile(file_name)
        
    
    def loadDataFile(self, file_name):
        
        print('reading:\t'+file_name.split('/')[-1])
        
        # the RW to SGM file triplet!
        files = [open(file_in) for file_in in [file_name, '.'.join(file_name.split('.')[:-2])+'.prblr.txt', '.'.join(file_name.split('.')[:-2])+'.outlr.txt']]
                
        # loop per sentence
        for lines in zip_longest(*files, fillvalue=''):

            # get the sentence (INPUT) and its PROBES and TARGETS
            line_input      = lines[0].split('|')[2].split()   

            line_probes     = [probe.split()[0] for probe in lines[1].split('|')[2:]]
            
            line_targets    = [target.strip().split('\t')[1:] for target in lines[2].split('|')[2:]]
            
            # avoid empty input, probes or targets lines
            if len(line_input) == 0 or len(line_probes) == 0 or len(line_targets) == 0:
                
                continue
            
            # get the sentence length and number of frames
            nr_words        = len(line_input)
            
            nr_frames       = max([int(target[1])+1 for target in line_targets])
            
            nr_probes       = len(line_targets)
            
            # filter sentences by length
            if nr_words in self.sentence_length and nr_probes in self.nr_probes and nr_frames <= self.max_nr_frames:

                line_input = [int(i) for i in line_input]
                    
                self.inputs.append(line_input)
                    
                self.targets.append(line_targets)
                    
                self.probes.append(line_probes)
                    
        for file in files: file.close()
    
    
    def encode_data(self, sentence_inputs, sentence_probes, sentence_targets):
        
        # add <BOS>
        sentence_inputs = [30001] + sentence_inputs
        
        #print(sentence_inputs, len(sentence_inputs))
        
        # the input batch
        #batch_ins = sentence_inputs + [0]*(self.max_nr_words-len(sentence_inputs))
        batch_ins = sentence_inputs + [0]*(self.max_nr_words+1-len(sentence_inputs))

        # the probe and rolefiller batches (empty)
        #batch_pbs = np.zeros((self.max_nr_words, self.nr_probes[-1], 300 + self.max_nr_frames + self.nr_tags), dtype=int)
        #batch_rfs = np.zeros((self.max_nr_words, self.nr_probes[-1], 300 + self.max_nr_frames + self.nr_tags), dtype=int)
        batch_pbs = np.zeros((self.max_nr_words+1, self.nr_probes[-1], 300 + self.max_nr_frames + self.nr_tags), dtype=int)
        batch_rfs = np.zeros((self.max_nr_words+1, self.nr_probes[-1], 300 + self.max_nr_frames + self.nr_tags), dtype=int)

        # fill up the probe and rolefiller sentence by sentence 
        nr_words_sentence  = len(sentence_inputs)
        nr_probes_sentence = len(sentence_probes)

        # probe by probe
        for p in range(0,nr_probes_sentence): 

            # the tag_id vector
            tag_id_vec = np.zeros((1, self.nr_tags), dtype=int)
            np.put(tag_id_vec, int(sentence_targets[p][0]), 1)
            
            # the frame_id vector
            frame_id_vec = np.zeros((1, self.max_nr_frames), dtype=int)
            np.put(frame_id_vec, int(sentence_targets[p][1]), 1)
                    
            # the word embedding vector
            word = str(sentence_targets[p][2])
            
            # if the word is in the embedding model, get the vector straight away,
            # otherwise try to capitalize it. If nothing works, assigne a zero-vector
            if self.emb_model:
                        
                if word in list(self.emb_model.index_to_key):
                        
                    word_emb = self.emb_model[word].reshape((1,300)).astype(int)
                    
                else:
                        
                    word = word.capitalize()
                        
                if word in list(self.emb_model.index_to_key):
                            
                    word_emb = self.emb_model[word].reshape((1,300)).astype(int)
                            
                else:
                        
                    word_emb = np.zeros((1,300), dtype=int)
                            
            else:
                        
                word_emb = np.zeros((1,300), dtype=int)
            
            # compose the probe vector
            if sentence_probes[p] == 'nan':
                        
                probe_vector = np.concatenate((tag_id_vec, frame_id_vec, np.zeros((1,300), dtype=int)), axis=1)
                        
            else:
                        
                probe_vector = np.concatenate((np.zeros((1, self.nr_tags), dtype=np.int), frame_id_vec, word_emb),axis=1)
            
            # compose the output vector
            target_vector = np.concatenate((tag_id_vec, frame_id_vec, word_emb),axis=1)
            
            # word by word
            for w in range(nr_words_sentence):
                
                batch_pbs[w,p,:] = probe_vector
                
                batch_rfs[w,p,:] = target_vector
        
        return Variable(torch.LongTensor(batch_ins)),Variable(torch.FloatTensor(batch_pbs)),Variable(torch.FloatTensor(batch_rfs))
        
        '''
        return torch.tensor(batch_ins, dtype=torch.int, requires_grad=False), torch.tensor(batch_pbs, dtype=torch.float, requires_grad=False), torch.tensor(batch_rfs, dtype=torch.float, requires_grad=False)
        '''
    
    