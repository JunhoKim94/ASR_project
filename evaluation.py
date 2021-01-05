from model.frontend import CustomFrontend
from model.model import ASRModel
from loader import *
import torch
from preprocess import *
import time
import argparse
from espnet.bin.asr_train import get_parser
from config import *
from tqdm import tqdm
import os
import pickle
from utils import *

device = torch.device("cuda:3")
#device = torch.device("cpu")
input_size = 80
#Hyper parameters
epochs = 80
batch_size = 8
SAMPLE_RATE = 16000
char = True

#with open("./split_data.pickle", "rb") as f:
#    a = pickle.load(f)

with open("./save_model/char2idx.pickle", "rb") as f:
    char2idx = pickle.load(f)

print(char2idx)
#test_path = a["test_path"]
#test_trg = a["test_trg"]
test_path = find_paths("./data/Test_Data")

test_loader = Batch_Loader(batch_size, device, test_path, 0, char2idx)

token_list = []
for key, value in char2idx.items():
    token_list.append(key)
token_list.append("<sos>")
vocab_size = len(token_list)
print(vocab_size)
#Transformer (Seq2Seq) 모델 --> 우리 Acoustic 모델

config = Config(token_list)
recog_config = Recog_config()
model = ASRModel(input_size = input_size,
                vocab_size = vocab_size,
                token_list = token_list,
                config = config,
                device = device)
model.to(device)
model.load_state_dict(torch.load("./save_model/best_ctc.pt", map_location = device))

eval_text(model, test_loader, recog_config, token_list, save_path = "./results/result_ctc_test.txt", char = char)

