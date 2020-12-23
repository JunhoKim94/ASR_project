import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wavio
import torch
import numpy as np

frame_length = 0.025
frame_stride = 0.010
N_FFT = 512
SAMPLE_RATE = 16000

def stft(filepath):
    
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()

    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)
    print(stft.shape)
    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    #feat = torch.FloatTensor(feat).transpose(0, 1)

    return feat

def Mel_S(filepath):

    # mel-spectrogram
    #sig, sr = librosa.load(filepath, sr = 16000)
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()
    sig = sig.astype(np.float32)
    #wav_length = len(y) / sr
    #sr means sampling rate
    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))
    S = librosa.feature.melspectrogram(y = sig, n_mels = 40, n_fft = input_nfft, hop_length = input_stride)
    mfcc = librosa.feature.mfcc(y = sig, sr = SAMPLE_RATE, n_mfcc = 40, n_fft = N_FFT, hop_length = input_stride)

    feat_mel = torch.FloatTensor(S)
    feat_mfcc = torch.FloatTensor(mfcc)
    feat_mel = feat_mel.unsqueeze(0)
    feat_mfcc = feat_mfcc.unsqueeze(0)
    feat = torch.cat([feat_mel,feat_mfcc],dim= 0)
    #S_po = librosa.power_to_db(feat, ref = np.max)
    return feat


def MFCC(wav_file):

    # mel-spectrogram
    #(rate, width, sig) = wavio.readwav(wav_file)
    #sig = sig.ravel()
    #sig = sig.astype(np.float32)
    sig, sr = librosa.load(wav_file, sr = 16000)
    wav_length = len(y) / sr
    #sr means sampling rate
    input_nfft = int(round(sr * frame_length))
    input_stride = len(y)//512 #int(round(sr * frame_stride))
    feat = librosa.feature.mfcc(y =sig, sr = sr, n_fft = input_nfft, n_mels = 26, hop_length = input_stride , n_mfcc = 40)
    print(feat.shape)
    feat[0] = librosa.feature.rmse(sig, hop_length = input_stride , frame_length = int(frame_length * SAMPLE_RATE))
    print(feat[0].shape)
    feat = [feat]
    feat.append(librosa.feature.delta(feat[0]))
    feat.append(librosa.feature.delta(feat[0], order = 2))
    return feat


if __name__ == "__main__":
    path = "./data/sample_data/성인남녀_001_A_001_M_KHI00_24_수도권_녹음실_00001.PCM"

    (rate, width, sig) = wavio.readwav(path)
    sig = sig.ravel()
    sig = sig.astype(np.float32)
    y, sr = librosa.load(path, sr = 16000)

    mel_x = Mel_S(path)
    mfcc_x = MFCC(path)

    print(mel_x.shape)
    for f in mfcc_x:
        print(f.shape)
    #print(mfcc_x.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)


    #ax1.pcolor(mfcc_x)
    ax1.pcolor(mfcc_x)
    #ax2.pcolor(mel_x)
    ax3.pcolor(x)
    #plt.pcolor(x)
    plt.show()