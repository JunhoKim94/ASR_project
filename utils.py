import torch
import random
import pickle
from tqdm import tqdm
import hgtk
import nltk
import Levenshtein as Lev 
import os
from hanspell import spell_checker

def char_distance(hyp, ref):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))
    #length = len(ref)

    return dist, length 

def split_path(path, trg, ratio = 0.1, save = False):
    
    train_path, train_trg = [], []
    val_path, val_trg = [], []
    test_path, test_trg = [], []
    for idx in range(len(path)):

        seed = random.random()

        if seed < 0.005:
            test_path.append(path[idx])
            test_trg.append(trg[idx])
        elif 0.005 < seed < ratio + 0.005:
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

def span_text(model, train_batch, recog_config, f, token_list, char):
    trg = train_batch["text"]
    ys_hat = model.recognize(train_batch["speech"], train_batch["speech_lengths"], recog_config)

    total_dist = 0
    total_length = 0
    for ys, tr in zip(ys_hat, trg):
        sen1, sen2 = "", ""
        for c in ys:
            if (c != -1) and (c != len(token_list) - 1):
                sen1 += token_list[c]
        for t in tr:
            if (t != -1) and (t != len(token_list) - 1):
                sen2 += token_list[t]

        if char:
            sen1, sen2 = hgtk.text.compose(sen1), hgtk.text.compose(sen2)
       
        #result = spell_checker.check(sen1)
        #result = result.as_dict()
        #f.write(sen1 + "\t")
        #sen1 = result["checked"]
        dist, length = char_distance(sen1, sen2)
        total_dist += dist
        total_length += length

        f.write(sen1 + "\t" + sen2)
        f.write("\n")

    return total_dist, total_length

def save_text(model, val_loader, recog_config, token_list, save_path = "./result.txt", char = True):
    score = 0
    f = open(save_path, "w")
    total_dist = 0
    total_length = 0

    total_size = len(val_loader) // val_loader.batch_size
    if len(val_loader) % val_loader.batch_size != 0:
        total_size += 1
    
    for i in tqdm(range(total_size)):
        val_batch = val_loader.get_batch(rand = False)
        dist, length = span_text(model, val_batch, recog_config, f, token_list, char)
        total_dist += dist
        total_length += length

    f.close()

    return total_dist / total_length


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



def eval_text(model, val_loader, recog_config, token_list, save_path = "./result.txt", char = True):
    score = 0
    f = open(save_path, "w")

    total_size = len(val_loader) // val_loader.batch_size
    if len(val_loader) % val_loader.batch_size != 0:
        total_size += 1
    
    for i in tqdm(range(total_size)):
        val_batch = val_loader.get_test_batch()
        batch_path = val_batch["path"]

        ys_hat = model.recognize(val_batch["speech"], val_batch["speech_lengths"], recog_config)

        for ys, p in zip(ys_hat, batch_path):
            sen1 = ""
            for c in ys:
                if (c != -1) and (c != len(token_list) - 1):
                    sen1 += token_list[c]

            if char:
                sen1 = hgtk.text.compose(sen1)
        
            f.write(p + "\t" + sen1 + "\n")

    f.close()

def find_paths(file_path = "./data/Test_Data"):
    x = os.listdir(file_path)
    ret_path = []
    for file in x:
        if len(file.split(".")) == 1:
            paths = os.listdir(file_path + "/" + file)
            for path in paths:
                if path.split(".")[-1].lower() == "pcm":
                    ret_path += [file_path + "/" + file + "/" + path]
        
    return ret_path

def get_distance(file_path):
    with open(file_path, "r") as f:
        x = f.readlines()

    total_dist = 0
    total_length = 0
    for line in x:
        s, t = line[:-1].split("\t")
        
        dist, length = char_distance(s, t)
        total_dist += dist
        total_length += length
    
    return total_dist / total_length