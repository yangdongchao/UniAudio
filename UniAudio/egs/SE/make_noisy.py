# This code aims to prepare SE training data, which use clean wave and nosie wave to simulate nosiy speech.
import os 
import random
import math
import csv
import argparse
import librosa
import soundfile as sf
import io
import numpy as np

import mmap
import os
import soundfile as sf

def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            data = mmap_o[offset : offset + length]
    return data

def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)

def get_parser():
    parser = argparse.ArgumentParser(
        description="helping to simulate speech enhancement data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clean-file", type=str, default=None, help="clean file in the format <exampe_id> <path>")
    parser.add_argument("--noise-file", type=str, default=None, help="noise file in the format <example_id> <path>")
    parser.add_argument("--noisy-path", type=str, default=None, help="the path to save noisy audio")
    parser.add_argument("--clean-path", type=str, default=None, help="the path to save noisy audio")
    parser.add_argument("--output-file", type=str, help="the output file, which records the noisy and clean information")
    return parser

def parse_filelist(file_path):
    '''
    解析txt路径文件
    '''
    files = []
    f = open(file_path,'r')
    for d in f.readlines():
        if d[-1] == "\n":
            files.append(d[:-1])
        else:
            files.append(d)
    return files

def load_audio_from_path(file_path):
    '''根据路径读取音频文件 '''
    '''
    audio_obj= wavio.read(file_path)
    audio = audio_obj.data[:,0].astype(np.float32)
    assert audio_obj.rate == 16000
    '''
    audio,sr = librosa.load(file_path, sr=16000)
    assert sr == 16000
    #print(audio_obj.sampwidth,audio_obj)
    #audio = wavfile.read(file_path)[1]
    #audio = audio / 32768 #2^15
    #audio = librosa.load(file_path,sr=16000)[0]
    #audio,fs= torchaudio.load(file_path,sr=16000)
    #assert fs == 16000
    #audio = audio.numpy()
    return audio

def sparse_chunk_file(file_path):
    ans = file_path.split(':')
    return ans[0], ans[1], ans[2]

def load_audio_from_chunks(file_path):
    ph, st, ed = sparse_chunk_file(file_path)
    byte_data = read_from_stored_zip(ph, int(st), int(ed))
    wav_path = io.BytesIO(byte_data)
    waveform, sample_rate = sf.read(wav_path, dtype="float32") # 
    return waveform

def split_wave_into2g(waveform):
    if waveform.shape[0] > 14*16000:
        waveform1 = waveform[:7*16000]
        waveform2 = waveform[7*16000:14*16000] # 
    else:
        waveform1 = waveform[:int(waveform.shape[0])//2]
        waveform2 = waveform[int(waveform.shape[0])//2:]
    return waveform1, waveform2

def save_audio_to_path(audio,store_path):
    '''将音频文件存储在给定路径下'''
    sf.write(store_path,audio, 16000,subtype='PCM_16')
    #wavfile.write(store_path, 16000, audio*32768)
    #wavio.write(store_path,audio,16000,sampwidth=2)

def SNR2scale(clean,noise,snr):
    clean_power = (clean**2).mean()
    noise_power = (noise**2).mean()
    
    scale = (
                10 ** (-snr / 20)
                * np.sqrt(clean_power)
                / np.sqrt(max(noise_power, 1e-10))
            )
    return scale

def mix_clean_noise_full(clean,noise,snr):
    '''
    给定纯净语音和噪声片段，基于一定的信噪比
    对clean和noise进行对齐并叠加
    '''
    mix_audio = np.zeros(clean.shape)

    print('mix:',mix_audio.shape,'noise:',noise.shape,'clean:',clean.shape)

    if clean.shape[0] >= noise.shape[0]:#如果纯净语音比带噪语音长
        offset = np.random.randint(0,2 * clean.shape[-1] - noise.shape[-1])
        # Repeat noise，先将噪声补齐至两倍长
        noise = np.pad(noise,[(offset, 2 * clean.shape[-1] - noise.shape[-1] - offset)],mode="wrap")
    #然后所有的流程都需要走这一分支
    #即：对于现在比原始音频长度长的噪声文件，随机选取start进行裁剪
    start = np.random.randint(0,noise.shape[-1]-clean.shape[-1])
    noise = noise[start:start+clean.shape[-1]]
    scale = SNR2scale(clean,noise,snr)#calculating after align the noise audio with the clean one
    noise = scale * noise
    #store the noise audio
    
    mix_audio = clean + noise
    return mix_audio,noise

def main_full_with_path():
    #write log in csv file
    args = get_parser().parse_args()
    mix_tr_rt = args.noisy_path # 
    clean_tr_rt = args.clean_path # 
    tr_csv = args.output_file
    f = open(tr_csv,'w',encoding = 'utf-8',newline = "")
    csv_write = csv.writer(f)
    csv_write.writerow(['clean','noise','snr','clean_path','noisy_path'])

    clean_rt = args.clean_file
    noise_rt = args.noise_file

    clean_lists = parse_filelist(clean_rt)
    noise_lists = parse_filelist(noise_rt)
    
    noise_nums = len(noise_lists)
 
    SNR_ends = [-20, 40]
    cnt = 0
    set_duration = 7
    #SNRs = [-20,-16,-12,-8,-4,0,4,8,12,16,20,24,28,32,36,40]
    for clean_path in clean_lists:
        clean_audio = load_audio_from_chunks(clean_path)
        #print('clean_audio ', clean_audio.shape)
        clean_audio1, clean_audio2 = split_wave_into2g(clean_audio)
        # print('clean_audio1, clean_audio2', clean_audio1.shape, clean_audio2.shape)
        # assert 1==2
        if clean_audio1.shape[0] < set_duration*16000:
            continue
        #clean_audio = load_audio_from_path(clean_path)
        noise_indexes = set()
        clean_audios = [clean_audio1, clean_audio2]
        for i, c_a in enumerate(clean_audios):
            noise_idx = random.randint(0,noise_nums-1)
            noise_path = noise_lists[noise_idx]
            noise = load_audio_from_path(noise_path)
            # print('noise ', noise.shape, c_a.shape)
            # assert 1==2
            n_zero = np.zeros(set_duration*16000)
            c_zero = np.zeros(set_duration*16000)
            if noise.shape[0] > set_duration*16000:
                idx = random.randint(0, noise.shape[0]-set_duration*16000-1)
                n_zero = noise[idx:idx+set_duration*16000] # 最多只用8s
            else:
                n_zero[:noise.shape[0]] = noise # 
            if c_a.shape[0] > set_duration*16000:
                idx = random.randint(0, c_a.shape[0]-set_duration*16000-1)
                c_zero = c_a[idx:idx + set_duration*16000]
            else:
                c_zero[:c_a.shape[0]] = c_a # 
            #select snr
            snr = random.randint(SNR_ends[0],SNR_ends[1])
            #snr = random.sample(SNRs,1)[0]
            #TODO:现在的mix函数返回两个值，noisy和noise
            mix_audio = mix_clean_noise_full(c_zero, n_zero, snr)[0]
            # print('mix_audio ', mix_audio.shape, c_zero.shape)
            # assert 1==2
            ph = clean_path.replace(':', '_') # transfer the : into _
            #sprint('ph ', ph, ph.split('/'))
            clean_name = ph.split('/')[-1] # get the last item
            print('clean_name ', clean_name)
            new_clean_name = clean_name + '_'+str(i) # get the order
            #clean_name和noise_name之间用'_'进行分隔
            mix_name = os.path.join(mix_tr_rt, new_clean_name + '_'+noise_path.split('/')[-1])
            print('mix_name:', mix_name)
            clean_name = os.path.join(clean_tr_rt, new_clean_name) + '.wav'
            #print('clean_name:', clean_name)
            save_audio_to_path(mix_audio, mix_name)
            save_audio_to_path(c_zero, clean_name)
            csv_write.writerow([clean_name[:-4], noise_path.split('/')[-1][:-4], snr, clean_name, mix_name])
            #print(clean_name[:-4],str(snr))
        #assert 1==2
    f.close()

if __name__ == "__main__":
    main_full_with_path()