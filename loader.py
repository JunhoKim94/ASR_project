from preprocess import *

class Data_Loader:
    def __init__(self, batch_size):

        self.batch_szie = batch_size

        char2idx = get_vocab()
        #print(char2idx)
        paths = get_path()
        data = get_channel_from_pcm(paths)

    def get_batch(self, rand = True):
        print(0)

    def padding(self, batch):
        '''
        data : list of data
        '''
        seq_length = 
        max_length = max(batch)

