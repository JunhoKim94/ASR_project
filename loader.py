from preprocess import *
import torch

class Data_Loader:
    def __init__(self, batch_size):

        self.batch_szie = batch_size

        char2idx = get_vocab()
        #print(char2idx)
        paths = get_path()
        data = get_channel_from_pcm(paths)

    def get_batch(self, rand = True):
        print(0)

    def padding(self, data, trg):
        '''
        data : list of data
        '''
        batch_size = len(data)
        channel = data[0].shape[1]

        seq_length = [len(b) for b in data]
        max_length = max(seq_length)
        
        trg_length = [len(b) for b in trg]
        max_trg = max(trg_length)

        torch_batch = torch.zeros(batch_size, max_length, channel)
        trg_batch = torch.zeros(batch_size, max_trg)

        for idx, (seq, t) in enumerate(zip(data, trg)):
            torch_batch[idx, :seq_length[idx], :] = seq
            trg_batch[idx, :trg_length[idx]] = t

        return torch_batch, trg_batch
