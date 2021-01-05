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
from torch.utils.tensorboard import SummaryWriter
import logging
logging.basicConfig(level=logging.ERROR)


writer = SummaryWriter(comment = "ASR-Transformers")

device = torch.device("cuda:2")
#device = torch.device("cpu")
input_size = 80
#Hyper parameters
epochs = 80
batch_size = 8
lr = 0.0025
warm_up = 8000
char = True

SAMPLE_RATE = 16000

path, trg, char2idx = preprocess_data(char = char)

ret = split_path(path, trg, 0.025, save = True)

train_path = ret["train_path"]
train_trg = ret["train_trg"]
val_path = ret["val_path"]
val_trg = ret["val_trg"]
test_path = ret["test_path"]
test_trg = ret["test_trg"]

dataloader = Batch_Loader(batch_size, device, train_path, train_trg, char2idx)
val_loader = Batch_Loader(batch_size, device, val_path, val_trg, char2idx)
test_loader = Batch_Loader(batch_size, device, test_path, test_trg, char2idx)
'''
dataloader = Batch_Loader(batch_size, device, val_path, val_trg, char2idx)
val_loader = Batch_Loader(batch_size, device, val_path, val_trg, char2idx)
test_loader = Batch_Loader(batch_size, device, test_path, test_trg, char2idx)
'''

token_list = []
for key, value in dataloader.char2idx.items():
    token_list.append(key)
token_list.append("<sos>")
vocab_size = len(token_list)
print(vocab_size)

config = Config(token_list)
recog_config = Recog_config()
model = ASRModel(input_size = input_size,
                vocab_size = vocab_size,
                token_list = token_list,
                config = config,
                device = device)


#model.to(device)
#model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

emb_size = config.adim

st = time.time()
total = len(dataloader) // batch_size + 1
best_acc = 0
step = 1
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    model.to(device)
    for iteration in tqdm(range(1, total)):

        for param_group in optimizer.param_groups:
            param_group['lr'] =  emb_size**(-0.5) * min(step**(-0.5), step * (warm_up**(-1.5)))
            lr = param_group['lr']
            step += 1


        train_batch = dataloader.get_batch()

        loss, ret_dict = model(**train_batch)
        acc = ret_dict["acc"]

        writer.add_scalar("loss/train", loss.item(), step)
        writer.add_scalar("acc/train", acc, step)

        epoch_loss += loss.item()
        epoch_acc += acc

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    epoch_loss /= total
    epoch_acc /= total

    if epoch % 4 == 0:
        current_time = round((time.time() - st) / 3600 , 4)
        cer, wer, val_acc = val_score(model, val_loader)
        #ys_hat = ret_dict["ys_hat"]
        #print(ys_hat[0], trg[0])
        writer.close()
        print(f"epoch : {epoch} | epoch loss : {epoch_loss} | acc : {epoch_acc} | val acc : {val_acc} | cer : {cer} | wer: {wer} | time : {current_time}")
        if best_acc < val_acc:
            best_acc = val_acc
            save_text(model, test_loader, recog_config, token_list, save_path = "./results/result_ctc.txt", char = char)
            torch.save(model.state_dict(), "./save_model/best_ctc.pt")

