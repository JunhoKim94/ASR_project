import numpy as np
import matplotlib.pyplot as plt
from model.frontend import CustomFrontend
import torch
from loader import *
import wave

def get_vocab():
    path = "./data/sample_data/label.txt"
    char2idx = dict()
    trg = []
    with open(path, "r", encoding = "UTF-8") as f:
        x = f.readlines()
        for sen in x:
            path, label = sen.split("\t")
            #label = label.replace(" ", "<blank>")
            trg.append(label[:-1])
            for c in label[:-1]:
                if c in char2idx:
                    continue
                char2idx[c] = len(char2idx)
       
    ret = []
    for sen in trg:
        temp = []
        for c in sen:
            temp.append(char2idx[c])
        ret.append(temp)
    
    return char2idx, ret


def get_path(folder_path = "./data/sample_data/"):
    paths = []
    with open("./data/sample_data/label.txt", "r", encoding = "UTF-8") as f:
        x = f.readlines()
        for sen in x:
            path, label = sen.split("\t")
            file_name = path.split("\\")[-1]
            path = folder_path + file_name
            paths.append(path)

    return paths


def get_channel_from_pcm(paths):
    stack = []
    for path in paths:
        with open(path, "rb") as f:
            buf = f.read()
            data = np.frombuffer(buf, dtype = 'int16')
            L = data[::2]
            R = data[1::2]
    
            temp = np.vstack([L,R])
            temp = torch.Tensor(temp).transpose(0,1)

            stack.append(temp)

    return stack

def plot_wav(path):
    with open(path, "rb") as f:
        buf = f.read()
        data = np.frombuffer(buf, dtype = 'int16')
        L = data[::2]
        R = data[1::2]

    print(L, R)
    print(len(L), len(R))
    sample_rate = 40800

    t = np.arange(0, 1., 1/sample_rate)
    fig, axs = plt.subplots (2,1)
    axs[0].plot(t, L[: len(t)])
    axs[1].plot(t, R[: len(t)])
    plt.savefig("./test.png")
    plt.show()        

if __name__ == "__main__":
    char2idx, trg = get_vocab()
    print(trg)
    #print(char2idx)
    '''
    paths = get_path()
    data = get_channel_from_pcm(paths)
    print(data[0].shape)
    data[1] = data[1].unsqueeze(0)

    print(data[1].shape)

    #print(data)
    frontend = CustomFrontend()
    a, length = frontend(data[1], torch.Tensor([data[1].shape[1], data[1].shape[1]]))
    print(a, length)
    print(a.shape, length.shape)
    '''
    wave.open("./data/sample_data/성인남녀_001_A_001_M_KHI00_24_수도권_녹음실_00001.PCM", "rb")