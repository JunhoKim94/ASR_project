import numpy as np
import matplotlib.pyplot as plt

def get_vocab():
    path = "./data/sample_data/label.txt"
    char2idx = dict()
    with open(path, "r") as f:
        x = f.readlines()
        for sen in x:
            path, label = sen.split("\t")

            for c in label[:-1]:
                if c in char2idx:
                    continue
                char2idx[c] = len(char2idx)
       
    return char2idx

def get_path():
    default = "./data/sample_data/"
    with open("./data/sample_data/label.txt", "r") as f:
        x = f.readlines()
        for sen in x:
            path, label = sen.split("\t")
            file_name = path.split("\\")[-1]
            path = default + file_name
            
            print(path)
#print(x)

if __name__ == "__main__":
    char2idx = get_vocab()
    print(char2idx)
    with open("./data/sample_data/성인남녀_001_A_001_M_KHI00_24_수도권_녹음실_00001.PCM", "rb") as f:
        buf = f.read ()
        data = np.frombuffer (buf, dtype = 'int16')
        L = data [:: 2]
        R = data [1 :: 2]

    print(L, R)
    print(len(L), len(R))
    sample_rate = 40800

    t = np.arange(0, 1., 1/sample_rate)
    fig, axs = plt.subplots (2,1)
    axs [0] .plot (t, L[: len(t)])
    axs [1] .plot (t, R[: len(t)])
    plt.savefig("./test.png")
    plt.show ()