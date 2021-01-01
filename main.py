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
from apex import amp

def adjust_learning_rate(optimizer, warmup_count):

    if warmup_count < 4752:
        lr = 1e-5 * (warmup_count / 4752)
    else:
        lr = 1e-5 * (1 - warmup_count / 47520)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


device = torch.device("cuda:2")
#device = torch.device("cpu")
input_size = 80
#Hyper parameters
epochs = 80
batch_size = 8
lr = 5e-4

SAMPLE_RATE = 16000


path, trg, char2idx = preprocess_data()
train_path, train_trg, val_path, val_trg,test_path, test_trg = split_path(path, trg, 0.025)

#Model 선언을 위해 필요한 모듈들
#vocab size(character 와 token 단위 중 선택해야 함.)
#dataloader = Data_Loader(batch_size, device)
#val_loader = Data_Loader(batch_size, device)
'''
dataloader = Batch_Loader(batch_size, device, train_path, train_trg, char2idx)
val_loader = Batch_Loader(batch_size, device, val_path, val_trg, char2idx)
test_loader = Batch_Loader(batch_size, device, test_path, test_trg, char2idx)
'''
dataloader = Batch_Loader(batch_size, device, val_path, val_trg, char2idx)
val_loader = Batch_Loader(batch_size, device, val_path, val_trg, char2idx)
test_loader = Batch_Loader(batch_size, device, test_path, test_trg, char2idx)


token_list = []
for key, value in dataloader.char2idx.items():
    token_list.append(key)
token_list.append("<sos>")
vocab_size = len(token_list)
print(vocab_size)
#enh = None
#전처리 과정을 담당하는 class
frontend = CustomFrontend(fs = 16000,
                            n_fft= 512,
                            normalized = True,
                            hop_length= int(0.01 * SAMPLE_RATE),
                            win_length= int(0.03 * SAMPLE_RATE), 
                            n_mels = 80)
#Data augmentation을 담당하는 class --> 시간적 비용이 많을 시 생략할것
#specaug = SpecAug()
#specaug.to(device)
specaug = None
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


#model.to(device)
#model = torch.nn.DataParallel(model)
model.to(device)
#save_text(model, test_loader, recog_config, token_list)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

st = time.time()
total = len(dataloader) // batch_size + 1
best_acc = 0
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    model.to(device)
    for iteration in tqdm(range(1, total)):
        train_batch = dataloader.get_batch()

        loss, ret_dict = model(**train_batch)
        acc = ret_dict["acc"]

        epoch_loss += loss.item()
        epoch_acc += acc

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    epoch_loss /= total
    epoch_acc /= total

    if epoch % 2 == 0:
        current_time = round((time.time() - st) / 3600 , 4)
        cer, wer, val_acc = val_score(model, val_loader)
        #ys_hat = ret_dict["ys_hat"]
        #print(ys_hat[0], trg[0])
        print(f"epoch : {epoch} | epoch loss : {epoch_loss} | acc : {epoch_acc} | val acc : {val_acc} | cer : {cer} | wer: {wer} | time : {current_time}")
        if best_acc < val_acc:
            best_acc = val_acc
            save_text(model, test_loader, recog_config, token_list, save_path = "result_apex.txt")
            torch.save(model.state_dict(), "./best_apex.pt")

