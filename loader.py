from preprocess import *
import torch
import random
from model.frontend import *

class Data_Loader:
    def __init__(self, batch_size, device):

        self.batch_size = batch_size
        self.device = device
        self.char2idx, self.trg = get_vocab()
        #print(char2idx)
        paths = get_path()
        self.data = get_channel_from_pcm(paths)
        self.idx = 0

    def __len__(self):
        return len(self.data)

    def get_seed(self, rand = True):
        if rand:
            seed = random.sample(range(len(self.data)), self.batch_size)
        
        else:
            max_idx = min((self.idx + 1) * self.batch_size, len(self.data))
            seed = [idx for idx in range(self.idx * self.batch_size, max_idx)]
            self.idx += 1

        return seed

    def get_batch(self, rand = True):
        seed = self.get_seed(rand)

        batch = [self.data[i] for i in seed]
        trg = [self.trg[i] for i in seed]
        batch, trg, seq_length, trg_length = self.padding(batch, trg)

        ret_dict = {"speech" : batch.to(self.device), 
                    "speech_lengths" : seq_length.to(self.device), 
                    "text" : trg.to(self.device), 
                    "text_lengths" : trg_length.to(self.device)}
        
        if (self.idx * self.batch_size) >= len(self.path):
            self.idx = 0
            
        return ret_dict
    def padding(self, data, trg):
        '''
        data : list of data
        '''
        batch_size = len(data)
        channel = data[0].shape[1]

        seq_length = torch.LongTensor([len(b) for b in data])
        max_length = max(seq_length)
        
        trg_length = torch.LongTensor([len(b) for b in trg])
        max_trg = max(trg_length)

        torch_batch = torch.zeros(batch_size, max_length, channel).fill_(-1)
        trg_batch = torch.zeros(batch_size, max_trg).to(torch.long).fill_(-1)

        for idx, (seq, t) in enumerate(zip(data, trg)):
            torch_batch[idx, :seq_length[idx], :] = seq
            trg_batch[idx, :trg_length[idx]] = torch.Tensor(t)

        return torch_batch, trg_batch, seq_length, trg_length

class Batch_Loader:
    def __init__(self, batch_size, device, path, trg, char2idx):

        self.batch_size = batch_size
        self.device = device
        self.path, self.trg, self.char2idx = path, trg, char2idx
        self.idx = 0

    def __len__(self):
        return len(self.path)

    def get_seed(self, rand = True):
        if rand:
            seed = random.sample(range(len(self.path)), self.batch_size)
        
        else:
            max_idx = min((self.idx + 1) * self.batch_size, len(self.path))
            seed = [idx for idx in range(self.idx * self.batch_size, max_idx)]
            self.idx += 1

        return seed

    def get_batch(self, rand = True):
        seed = self.get_seed(rand)

        batch_paths = [self.path[i] for i in seed]
        trg = [self.trg[i] for i in seed]
        batch, trg, seq_length, trg_length = self.padding(batch_paths, trg)

        ret_dict = {"speech" : batch.to(self.device), 
                    "speech_lengths" : seq_length.to(self.device), 
                    "text" : trg.to(self.device), 
                    "text_lengths" : trg_length.to(self.device)}

        if (self.idx * self.batch_size) >= len(self.path):
            self.idx = 0

        return ret_dict

    
    def padding(self, paths, trg):
        '''
        data : list of data
        '''

        data = get_channel_from_pcm(paths)
        batch_size = len(data)
        channel = data[0].shape[1]

        seq_length = torch.LongTensor([len(b) for b in data])
        max_length = max(seq_length)
        
        trg_length = torch.LongTensor([len(b) for b in trg])
        max_trg = max(trg_length)

        torch_batch = torch.zeros(batch_size, max_length, channel).fill_(-1)
        trg_batch = torch.zeros(batch_size, max_trg).to(torch.long).fill_(-1)

        for idx, (seq, t) in enumerate(zip(data, trg)):
            torch_batch[idx, :seq_length[idx], :] = seq
            trg_batch[idx, :trg_length[idx]] = torch.Tensor(t)

        return torch_batch, trg_batch, seq_length, trg_length



if __name__ == "__main__":
    dataloader = Batch_Loader(1000, torch.device("cpu"))
    a = dataloader.get_batch()
    print(a)