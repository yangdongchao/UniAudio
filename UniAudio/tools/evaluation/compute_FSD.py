# Copyright 2023
# Author     : UniAudio Teams
# Description: Following Meta's VoiceBox. Fréchet Speech Distance (FSD). using Wave2Vec2.0 models to extract deep features

import os
import glob
import argparse
from tqdm import tqdm
from scipy.io import wavfile
from pystoi import stoi
import numpy as np
import scipy.linalg
import soundfile as sf
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()    

def get_features(ref_wav):
    audio_input, sample_rate = librosa.load(ref_wav, sr=16000)  # (31129,)
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values  # torch.Size([1, 31129])
    with torch.no_grad():
        frame_level_features = model(input_values.cuda())['last_hidden_state'] 
    return frame_level_features.mean(1).detach().cpu() # (1, 768)

# small fsd indicates better performance
def calculate_fsd(features_1, features_2): # using 2048 layer to calculate
    eps = 1e-6
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    stat_1 = { # 计算均值和协方差矩阵
        'mu': np.mean(features_1.numpy(), axis=0),
        'sigma': np.cov(features_1.numpy(), rowvar=False),
    }
    stat_2 = {
        'mu': np.mean(features_2.numpy(), axis=0),
        'sigma': np.cov(features_2.numpy(), rowvar=False),
    }
    print('Computing Frechet Speech Distance')
    mu1, sigma1 = stat_1['mu'], stat_1['sigma']
    mu2, sigma2 = stat_2['mu'], stat_2['sigma']
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'WARNING: fid calculation produces singular product; adding {eps} to diagonal of cov')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            assert False, 'Imaginary component {}'.format(m)
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return {
        'frechet_speech_distance': float(fid),
    }

def get_fsd(ref_dir, deg_dir, num_generated=5):
    input_files = glob.glob(f"{ref_dir}/*.wav")
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {ref_dir}")
    #inputs = feature_extractor(audio, padding=True, return_tensors="pt")
    ref_features = []
    deg_features = []
    for ref_wav in tqdm(input_files):
        ref_feature = get_features(ref_wav) # get reference features
        ref_features.append(ref_feature)
        #for j in range(num_generated):
        deg_wav = os.path.join(deg_dir, os.path.basename(ref_wav)) # [:-4]+'_'+str(j)+'.wav'
        deg_feature = get_features(deg_wav)
        deg_features.append(deg_feature)
    ref_features = torch.cat(ref_features, dim=0) # B, 768
    deg_features = torch.cat(deg_features, dim=0) # B, 768
    fsd = calculate_fsd(ref_features, deg_features)
    return fsd['frechet_speech_distance']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute speaker similarity_score")
    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )
    args = parser.parse_args()
    similarity_score = get_fsd(args.ref_dir, args.deg_dir)
    print(f"FSD: {similarity_score}")
