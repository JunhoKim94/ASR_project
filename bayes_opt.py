import numpy as np
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt import gp_minimize
import math
import pickle
import logging
from config import *
logging.basicConfig(level=logging.ERROR)
import torch
from loader import *
from model.model import ASRModel

device = torch.device("cuda:3")
#device = torch.device("cpu")
input_size = 80
batch_size = 4
char = True
#Hyper parameters
with open("./save_model/split_data.pickle", "rb") as f:
    a = pickle.load(f)

with open("./save_model/char2idx.pickle", "rb") as f:
    char2idx = pickle.load(f)

print(char2idx)
test_path = a["test_path"]
test_trg = a["test_trg"]

test_loader = Batch_Loader(batch_size, device, test_path, test_trg, char2idx)

token_list = []
for key, value in char2idx.items():
    token_list.append(key)
token_list.append("<sos>")
vocab_size = len(token_list)
print(vocab_size)
#Transformer (Seq2Seq) 모델 --> 우리 Acoustic 모델
config = Config(token_list)
model = ASRModel(input_size = input_size,
                vocab_size = vocab_size,
                token_list = token_list,
                config = config,
                device = device)
model.to(device)
model.load_state_dict(torch.load("./save_model/best_ctc.pt", map_location = device))

#dim_freeze = Categorical(2)


dim_ctc = Real(low = 0.1, high = 0.5, name = "ctc")
dim_beam = Integer(low = 2, high = 5, name = "beam")
dim_penalty = Real(low = 0.4, high = 2.0, name = "penalty")

dimensions = [dim_ctc, dim_beam, dim_penalty]
default_parameters = [0.3, 3, 1.2]

@use_named_args(dimensions = dimensions)
def fitness(ctc, beam, penalty):

    print(f"ctc : {ctc} | beam : {beam} | penalty : {penalty}")
    recog_config = Recog_config()
    recog_config.ctc_weight = ctc
    recog_config.beam_size = beam
    recog_config.penalty = penalty

    score = save_text(model, test_loader, recog_config, token_list, save_path = "./results/result_ctc_test.txt", char = char)
    print(score)
    return -score

search_result = gp_minimize(func = fitness, 
                            dimensions = dimensions,
                            acq_func = "EI",
                            n_calls = 12,
                            x0 = default_parameters)

print(search_result.x)

with open("./eff_ctc_0.3.pickle", "wb") as f:
    pickle.dump(search_result, f)