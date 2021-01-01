from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet2.asr.espnet_joint_model import ESPnetEnhASRModel
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.ctc import CTC
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr.frontend.default import DefaultFrontend

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

device = torch.device("cpu")
#device = torch.device("cpu")
input_size = 80
#Hyper parameters
epochs = 80
batch_size = 8
lr = 1e-3

with open("./train.pickle", "rb") as f:
    a = pickle.load(f)

path, trg, char2idx = preprocess_data()
#ret = {"train_path" : train_path, "train_trg" : train_trg, "val_path" : val_path, "val_trg" : val_trg ,"test_path" : test_path, "test_trg" : test_trg}
test_path = a["val_path"]
test_trg = a["val_trg"]
test_loader = Batch_Loader(batch_size, device, test_path, test_trg, char2idx)

token_list = []
for key, value in char2idx.items():
    token_list.append(key)
token_list.append("<sos>")
vocab_size = len(token_list)
print(vocab_size)
#enh = None
#전처리 과정을 담당하는 class
frontend = CustomFrontend(fs = 16000,
                            n_fft= 512,
                            normalized = True,
                            n_mels = 80)
#Data augmentation을 담당하는 class --> 시간적 비용이 많을 시 생략할것
specaug = SpecAug()
#specaug.to(device)
#specaug = None
#전처리 후 데이터 normalize를 담당하는 class
normalize = None
#Transformer (Seq2Seq) 모델 --> 우리 Acoustic 모델

config = Config(token_list)
recog_config = Recog_config()
model = ASRModel(input_size = input_size,
                vocab_size = vocab_size,
                token_list = token_list,
                frontend = frontend,
                specaug = specaug,
                normalize = normalize,
                config = config)

model.load_state_dict(torch.load("./best.pt", map_location = device))

save_text(model, test_loader, recog_config, token_list, save_path = "./result_eval.txt")