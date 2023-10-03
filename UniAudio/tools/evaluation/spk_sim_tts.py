import soundfile as sf
import torch
#import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models import ECAPA_TDNN_SMALL
import argparse
import glob
from tqdm import tqdm
import os
import numpy as np
MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def verification_case(model, wav1, wav2, use_gpu=True):
    wav1, sr1 = sf.read(wav1)
    wav2, sr2 = sf.read(wav2)

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    if use_gpu:
        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    return sim


def verification(model_name,  ref_dir, deg_dir, use_gpu=True, checkpoint=None):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint)
    
    if use_gpu:
        model = model.cuda()
    deg_files = glob.glob(f"{deg_dir}/*.wav")
    if len(deg_files) < 1:
        raise RuntimeError(f"Found no wavs in {deg_dir}")
    #print(deg_files)
    similarity_scores = []
    for deg_wav in tqdm(deg_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav)) # 
        if os.path.exists(ref_wav) == False:
            continue
        sim = verification_case(model, deg_wav, ref_wav, use_gpu)
        similarity_scores.append(abs(sim[0].item()))
    
    print(np.mean(similarity_scores))
    #print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))


if __name__ == "__main__":
    #fire.Fire(verification)
    parser = argparse.ArgumentParser(description="Compute speaker similarity_score")
    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder or file list."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )
    args = parser.parse_args()
    verification(model_name='wavlm_large', ref_dir=args.ref_dir, deg_dir=args.deg_dir,
                 checkpoint='checkpoint/wavlm_large_finetune.pth')
    