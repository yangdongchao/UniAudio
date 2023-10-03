#!/usr/bin/env bash

# pip install pesq
# pip install pystoi
# pip install pyworld
# pip install pysptk
# pip install -U numpy
stage=6
stop_stage=6
ref_dir=''
gen_dir=''
work_place=''
echo ${ref_dir}
echo ${gen_dir}

if [ $stage -le 1 ] && [ "${stop_stage}" -ge 1 ];then
  echo "Compute SSIM"
  python compute_ssim.py \
    -r ${ref_dir} \
    -d ${gen_dir}
fi

if [ $stage -le 2 ] && [ "${stop_stage}" -ge 2 ];then
  echo "Compute PESQ"
  python compute_pesq.py \
    -r ${ref_dir} \
    -d ${gen_dir}
fi

if [ $stage -le 3 ] && [ "${stop_stage}" -ge 3 ];then
  echo "Compute STOI"
  python compute_stoi.py \
    -r ${ref_dir} \
    -d ${gen_dir}
fi

if [ $stage -le 4 ] && [ "${stop_stage}" -ge 4 ];then
  echo "Compute MS-STFT-Loss"
  python compute_ms_stft_loss.py \
    -r ${ref_dir} \
    -d ${gen_dir} \
    -s 16000
fi

if [ $stage -le 5 ] && [ "${stop_stage}" -ge 5 ];then
  echo "Compute SI-SNR"
  python compute_sisnr.py \
    -r ${ref_dir} \
    -d ${gen_dir} 
fi

if [ $stage -le 6 ] && [ "${stop_stage}" -ge 6 ];then
  echo "Compute MCD"
  mkdir ${work_place}/mel_r
  mkdir ${work_place}/mel_d
  python compute_mcd.py \
    -r ${ref_dir} \
    -d ${gen_dir} \
    --mel-d ${work_place}/mel_d \
    --mel-r ${work_place}/mel_r \
    -s 16000
  
  rm -rf ${work_place}/mel_r
  rm -rf ${work_place}/mel_d
fi

if [ $stage -le 7 ] && [ "${stop_stage}" -ge 7 ];then
    echo "Compute speaker similarity" # follow valle and voicebox to cal SIM 
    python spk_sim_tts.py \
    -r ${ref_dir} \
    -d ${gen_dir} 
fi

if [ $stage -le 8 ] && [ "${stop_stage}" -ge 8 ];then
    echo "Compute FSD"
    python compute_FSD.py \
    -r ${ref_dir} \
    -d ${gen_dir} 
fi


if [ $stage -le 9 ] && [ "${stop_stage}" -ge 9 ];then
    echo "Compute CDPAm"
    python compute_cdpam.py \
    -r ${ref_dir} \
    -d ${gen_dir} 
fi

if [ $stage -le 10 ] && [ "${stop_stage}" -ge 10 ];then
    echo "Compute visqol"
    python compute_visqol.py \
    -r ${ref_dir} \
    -d ${gen_dir} 
fi

if [ $stage -le 11 ] && [ "${stop_stage}" -ge 11 ];then
    echo "Compute fad_kl"
    python compute_fad_kl.py \
    -r ${ref_dir} \
    -d ${gen_dir} 
fi

if [ $stage -le 12 ] && [ "${stop_stage}" -ge 12 ];then
    echo "Compute clap score"
    python compute_clap_score2.py \
    -a ${audio_file} \
    -t ${text_file} 
fi

if [ $stage -le 13 ] && [ "${stop_stage}" -ge 13 ];then
    echo "Compute speaker similarity" # follow make-a-voice to calculate sim for VC
    python compute_similarity_vc.py \
    -r ${ref_dir} \
    -d ${gen_dir} 
fi