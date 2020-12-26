from preprocess import *
import torch
import random
from model.frontend import *

class Data_Loader:
    def __init__(self, batch_size):

        self.batch_size = batch_size
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

        return batch, trg, seq_length, trg_length

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

        torch_batch = torch.zeros(batch_size, max_length, channel)
        trg_batch = torch.zeros(batch_size, max_trg).to(torch.long)

        for idx, (seq, t) in enumerate(zip(data, trg)):
            torch_batch[idx, :seq_length[idx], :] = seq
            trg_batch[idx, :trg_length[idx]] = torch.Tensor(t)

        return torch_batch, trg_batch, seq_length, trg_length

if __name__ == "__main__":
    dataloader = Data_Loader(10)
    
    batch, trg, seq_length, trg_length = dataloader.get_batch()

    print(batch.shape, trg.shape)
    frontend = CustomFrontend()
    a, length = frontend(batch, seq_length)
    #print(a, length)
    print(a.shape, length.shape)