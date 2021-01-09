from model.model import ASRModel
from loader import *
import torch
import time
import argparse
from config import *
from tqdm import tqdm
import os
import pickle
from utils import *
import logging
logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type = str, help = "Test할 파일이 있는 폴더 경로를 입력하세요")
parser.add_argument("--output_dir", type = str, help = "출력 파일 경로를 입력하세요")
args = parser.parse_args()

print(f"Your test folder is {args.input_dir}")
print(f"Your output file is {args.output_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 80
#Hyper parameters
batch_size = 8
char = True

with open("./save_model/split_data.pickle", "rb") as f:
    a = pickle.load(f)
test_path = a["test_path"]
test_trg = a["test_trg"]


with open("./save_model/char2idx.pickle", "rb") as f:
    char2idx = pickle.load(f)


print(char2idx)
#test_path = find_paths(args.input_dir)
#test_trg = None

test_loader = Batch_Loader(batch_size, device, test_path, test_trg, char2idx)

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

if test_trg == None:
    eval_text(model, test_loader, recog_config, token_list, save_path = args.output_dir, char = char)
else:
    score = save_text(model, test_loader, recog_config, token_list, save_path = "./results/result_ctc_eval.txt", char = char)
    print((1 - score) * 100)
