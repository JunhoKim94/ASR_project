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


#device = torch.device("cuda:1")
device = torch.device("cpu")
input_size = 80
#Hyper parameters
epochs = 80
batch_size = 16
lr = 1e-3


#Model 선언을 위해 필요한 모듈들
#vocab size(character 와 token 단위 중 선택해야 함.)
dataloader = Data_Loader(batch_size, device)
token_list = []
for key, value in dataloader.char2idx.items():
    token_list.append(key)
token_list.append("<sos>")
vocab_size = len(token_list)
#enh = None
#전처리 과정을 담당하는 class
frontend = CustomFrontend(fs = 16000,
                            n_fft= 512,
                            normalized = True,
                            n_mels = 80)
frontend.to(device)
#Data augmentation을 담당하는 class --> 시간적 비용이 많을 시 생략할것
#specaug = SpecAug()
#specaug.to(device)
specaug = None
#전처리 후 데이터 normalize를 담당하는 class
normalize = None
#Transformer (Seq2Seq) 모델 --> 우리 Acoustic 모델

encoder =  TransformerEncoder(input_size = input_size, 
                                padding_idx = -1, 
                                attention_heads = 8, 
                                output_size = 512)
decoder = TransformerDecoder(vocab_size = vocab_size, 
                            encoder_output_size = 512,
                            attention_heads = 8
                            )
encoder.to(device)
decoder.to(device)
#CTC loss added
ctc = CTC(vocab_size, 512, dropout_rate = 0.2, ignore_nan_grad = True)
ctc.to(device)
#전체 model binding ESPnet
model = ASRModel(vocab_size = vocab_size, 
                            encoder = encoder, 
                            decoder = decoder, 
                            ctc = ctc, 
                            token_list = token_list,  
                            frontend = frontend, 
                            specaug = specaug, 
                            ctc_weight = 0.3,
                            ignore_id = -1,
                            normalize = normalize,
                            rnnt_decoder = None,
                            sym_space= " ",
                            sym_blank= "-")

#model = E2E(idim = 80, odim = vocab_size, args)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

print(model.training)
st = time.time()
total = len(dataloader) // batch_size + 1
for epoch in range(epochs):
    model.training = False
    epoch_loss = 0
    epoch_acc = 0
    epoch_cer = 0
    epoch_wer = 0
    for iteration in range(1, total):
        train_batch = dataloader.get_batch()

        loss, ret_dict, _ = model(**train_batch)
        test = model.recognize(train_batch["speech"], train_batch["speech_lengths"])
        print(test)
        
        epoch_loss += loss.item()
        epoch_acc += ret_dict["acc"].item()
        epoch_cer += ret_dict["cer"].item()
        epoch_wer += ret_dict["wer"].item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    epoch_loss /= total
    epoch_acc /= total
    epoch_cer /= total
    epoch_wer /= total

    if epoch % 2 == 0:
        current_time = round((time.time() - st) / 3600 , 4)
        ys_hat = ret_dict["ys_hat"]
        trg = train_batch["text"]
        temp1, temp2 = "", ""
        for c, t in zip(ys_hat[0], trg[0]):
            #print(token_list[c], token_list[t])
            if c != -1:
                temp1 += token_list[c]
            if t != -1:
                temp2 += token_list[t]

        print(temp1, temp2)
        print(ys_hat[0], trg[0])
        print(f"epoch : {epoch} | epoch loss : {epoch_loss} | acc : {epoch_acc} | cer : {epoch_cer} | wer: {epoch_wer} | time : {current_time}")