import numpy as np
import matplotlib.pyplot as plt
from model.frontend import CustomFrontend
import torch
from loader import *
import wave
import os
import hgtk
from utils import *

def get_vocab():
    path = "./data/sample_data/label.txt"
    char2idx = {"PAD" : -1, "-" : 0}
    trg = []
    with open(path, "r", encoding = "UTF-8") as f:
        x = f.readlines()
        for sen in x:
            path, label = sen.split("\t")
            #label = label.replace(" ", "<space>")
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

def preprocess_data(char = True):
    paths = ["./data/2020_자유대화_Hackarthon_학습DB/001.일반남녀/000.PCM2TEXT/2020_일반남녀_학습DB_PCM2TEXT.txt",
            "./data/2020_자유대화_Hackarthon_학습DB/002.노인남녀(시니어)/000.PCM2TEXT/2020_시니어_학습DB_PCM2TEXT.txt",
            "./data/2020_자유대화_Hackarthon_학습DB/003.소아남녀/000.PCM2TEXT/2020_소아남녀_학습DB_PCM2TEXT.txt",
            "./data/2020_자유대화_Hackarthon_학습DB/004.외래어/000.PCM2TEXT/2020_외래어_학습DB_PCM2TEXT.txt"]
    ret_paths = []
    trg = []
    char2idx = {"PAD" : -1, "-" : 0}
    delete = 0
    for path in paths:
        with open(path, "r", encoding= "UTF-8") as f:
            x = f.readlines()
            for sen in x:
                p, label = sen.split("\t")
                p = p.split("\\")
                if p[3] == "성인남녀_002_C_030_F_OSS00_44_충청_녹음실":
                    a = p[4].split(".")
                    p[4] = a[0] + ".pcm"

                real_path = f"./data/{p[1]}/{p[2]}/{p[3]}/{p[4]}"
                if not os.path.isfile(real_path):
                    delete += 1
                    continue
                
                if char:
                    label = hgtk.text.decompose(label)

                ret_paths.append(real_path)
                trg.append(label[:-1])
                
                for c in label[:-1]:
                    if c in char2idx:
                        continue
                    char2idx[c] = len(char2idx)

    print(char2idx)
    ret_trg = []
    for sen in trg:
        ret_trg.append([char2idx[c] for c in sen])

    print(delete, "paths are deleted")
    print(len(ret_paths) ,"data preprocessed")

    with open("./char2idx.pickle", "wb") as f:
        pickle.dump(char2idx, f)

    split_path(ret_paths, ret_trg, ratio = 0.05)


    return ret_paths, ret_trg, char2idx


def get_channel_from_pcm(paths):
    stack = []
    for path in paths:
        with open(path, "rb") as f:
            buf = f.read()
            data = np.frombuffer(buf, dtype = 'int16')
            #L = data[::2]
            #R = data[1::2]
            #print(data.shape)
            temp = torch.Tensor(data).unsqueeze(1)
            #temp = np.vstack([L,R])
            #temp = torch.Tensor(temp).transpose(0,1)

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
    sample_rate = 16000

    t = np.arange(0, 1., 1/sample_rate)
    fig, axs = plt.subplots (2,1)
    axs[0].plot(t, L[: len(t)])
    axs[1].plot(t, R[: len(t)])
    plt.savefig("./test.png")
    plt.show()        

if __name__ == "__main__":

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
    
