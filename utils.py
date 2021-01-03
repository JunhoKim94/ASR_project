import torch
import random
import pickle
from tqdm import tqdm
import hgtk
import nltk
import Levenshtein as edit

def split_path(path, trg, ratio = 0.1, save = False):
    
    train_path, train_trg = [], []
    val_path, val_trg = [], []
    test_path, test_trg = [], []
    for idx in range(len(path)):

        seed = random.random()

        if seed < 0.001:
            test_path.append(path[idx])
            test_trg.append(trg[idx])
        elif 0.001 < seed < ratio + 0.001:
            val_path.append(path[idx])
            val_trg.append(trg[idx])
        else:
            train_path.append(path[idx])
            train_trg.append(trg[idx])
    
    ret = {"train_path" : train_path, "train_trg" : train_trg, "val_path" : val_path, "val_trg" : val_trg ,"test_path" : test_path, "test_trg" : test_trg}
    if save:
        with open("./split_data.pickle","wb") as f:
            pickle.dump(ret, f)

    return ret

def span_text(model, train_batch, recog_config, f, token_list):
    trg = train_batch["text"]
    ys_hat = model.recognize(train_batch["speech"], train_batch["speech_lengths"], recog_config)

    score = 0
    for ys, tr in zip(ys_hat, trg):
        sen1, sen2 = "", ""
        for c in ys:
            #print(token_list[c], token_list[t])
            if (c != -1) and (c != token_list[-1]):
                sen1 += token_list[c]
        for t in tr:
            if (t != -1) and (t != token_list[-1]):
                sen2 += token_list[t]

        sen1, sen2 = hgtk.text.compose(sen1), hgtk.text.compose(sen2)
       
        score += edit.distance(sen1, sen2) / len(sen2)

        f.write(sen1 + "\t" + sen2)
        f.write("\n")

    return score / len(trg)

def save_text(model, val_loader, recog_config, token_list, save_path = "./result.txt"):
    device = torch.device("cpu")
    model.to(device)
    val_loader.device = device
    score = 0
    f = open(save_path, "w")
    for i in tqdm(range(len(val_loader) // val_loader.batch_size)):
        val_batch = val_loader.get_batch(rand = False)
        s = span_text(model, val_batch, recog_config, f, token_list)
        score += s
        print(s)
    f.close()

    return score


def val_score(model, val_loader):
    
    model.eval()
    val_cer = 0
    val_acc = 0
    val_wer = 0
    for i in range(len(val_loader)):
        val_batch = val_loader.get_batch(rand = False)

        loss, ret_dict = model(**val_batch)

        val_cer += ret_dict["cer"]
        val_wer += ret_dict["wer"]
        val_acc += ret_dict["acc"]
    
    return val_cer / len(val_loader) , val_wer / len(val_loader), val_acc / len(val_loader)
