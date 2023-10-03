
# A demo recipe for plain_tts based on LibriTTS dataset.
# plain_tts: phone ---> wave
. ./path.sh

pip3 install fairseq==0.12.2 einops==0.6.0 sentencepiece encodec

stage=1
stop_stage=100
ngpu=1  # how many GPUs, you want to use to train the model

train_set="train"
valid_set="val"
test_sets=""

# training config
seed=999
debug=false
batch_scale=8000 # the total number of tokens in one batch
learning_rate=0.005 # the learning rate
port=12345
train_opts=
inference_opts=
tag=
inference_tag=default
resume=
data_tag=
TASK='plain_tts'

if [ ! -d "utils" ]; then
  ln -s ../tools/kaldi/utils ./
fi
if [ ! -d "data_scripts" ]; then
  ln -s ../tools/data_scripts ./
fi

. utils/parse_options.sh

if [ ! -z $resume ]; then
    train_opts="--resume $resume"
    inference_opts="--resume $resume"
fi

if [ $debug == true ]; then
    export HOST_GPU_NUM=1
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"

else
    export HOST_GPU_NUM=8
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="localhost"
    train_opts="$train_opts"
fi

### stage 1-3: data preparation ###

# Prepare data following Espnet and split
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare LibriTTS dataset"
    # this part aims to get the information about the dataset. 
    # Considering different tasks using different dataset, we donot provide the scripts to access dataset
    # for audio data, please prepare wav.scp 
    # for phone data, please prepare phone.scp
    # please go to ../data_samples folder to see the samples
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "split the data for $ngpu GPUs"

    for part in $test_sets $valid_set $train_set; do
      mkdir -p data/${part}/${ngpu}splits
      # extra shuf to ensure balance across GPUs
      # So the generated data cannot be reproduced due to the shuffle randomness
      cat data/${part}/wav.scp | shuf >  data/${part}/wav.scp.shuf
      split_scp=
      for n in `seq 1 $ngpu`; do
          split_scp="$split_scp data/${part}/${ngpu}splits/wav.${n}.scp"
      done
      utils/split_scp.pl data/${part}/wav.scp.shuf $split_scp

      # split the text accordingly
      for n in `seq 1 $ngpu`; do
          python3 data_scripts/filter_scp.py \
            data/${part}/${ngpu}splits/wav.${n}.scp \
            data/${part}/phone.scp \
            data/${part}/${ngpu}splits/phone.${n} &
      done; wait


    done
fi

# Plain TTS requires 2 data keys: phone_seq, audio_seq
# stage 2-3 process audio_seq and phone_seq respectively
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Prepare text and audio sequence"
    for part in $valid_set $train_set; do
    # for part in $valid_set; do
      echo "prepare $part ... "

      # phone alignment obtained by Kaldi
      utils/run.pl JOB=1:${ngpu} data/${part}/${ngpu}splits/log/g2p_offline.JOB.log \
        python3 data_scripts/offline_tokenization.py \
          --input-file data/${part}/${ngpu}splits/phone.JOB \
          --output-file data/${part}/${ngpu}splits/alignment.JOB.pt \
          --tokenizer alignment --rank JOB || exit 1;

      # Prompt
      utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/filter_utt2spk.JOB.log \
        python3 data_scripts/filter_scp.py \
          data/${part}/${ngpu}splits/wav.JOB.scp data/${part}/utt2spk \
          data/${part}/${ngpu}splits/utt2spk.JOB || exit 1;

      # Audio
      utils/run.pl JOB=1:$ngpu data/${part}/${ngpu}splits/log/audio_codec_dump.JOB.log \
        python3 data_scripts/offline_tokenization.py \
          --input-file data/${part}/${ngpu}splits/wav.JOB.scp \
          --output-file data/${part}/${ngpu}splits/audio_codec.JOB.pt \
          --tokenizer audio --rank JOB || exit 1;
      
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "create data json"
    # json index from 0 but all data files index from 1
    for part in $valid_set $train_set; do
      for n in `seq 0 $[$ngpu-1]`; do
        # TTS
        python3 data_scripts/create_data_json.py \
         --task tts \
         --out-json   $PWD/data/${part}/${ngpu}splits/data_tts.${n}.json \
         --prompt_seq $PWD/data/${part}/${ngpu}splits/utt2spk.$[$n+1] \
         --phone_seq  $PWD/data/${part}/${ngpu}splits/alignment.$[$n+1].pt \
         --audio_seq  $PWD/data/${part}/${ngpu}splits/audio_codec.$[$n+1].pt \
         & 
      done; wait

    done
fi

### Stage 4: Training ###
if [ -z $data_tag ] && [ $stop_stage -le 4 ]; then
    echo "you should provide data tag" || exit 1;
fi

train_data_jsons="data/${train_set}/${ngpu}splits/data_tts.ALL.json"
valid_data_jsons="data/${valid_set}/${ngpu}splits/data_tts.ALL.json"

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    mkdir -p exp 
    if [ -z $tag ]; then
        echo "please provide a tag for this experiment" && exit 1;
    fi
    echo "stage 5: training..."
    NCCL_DEBUG=TRACE torchrun \
        --nproc_per_node ${HOST_GPU_NUM} --master_port $port \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        ../../train.py \
        --exp_dir exp \
        --seed $seed \
        --cudnn_deterministic \
        --train_data_jsons $train_data_jsons \
        --valid_data_jsons $valid_data_jsons \
        --batch_scale $batch_scale \
        --learning_rate $learning_rate \
        --non-acoustic-repeat 3 \
        --audio-tokenizer "soundstream" \
        --audio-prompt-tokenizer "audio_prompt" \
        --phone-tokenizer "alignment" \
        --semantic-tokenizer "hubert" \
        --semantic-tokenizer-duplicate true \
        --singPhoneTokenizer "sing_phone" \
        --singMidiTokenizer "sing_midi" \
        --FrozenT5Embedder "text_t5" \
        --n_layer 24 \
        --n_head 16 \
        --n_embd 1536 \
        $train_opts
fi

# TTS inference
vc_test_sets="libritts_test"
inference_tag="tts_inference"
inference_dir=exp/${tag}/inference_${inference_tag}
ngpu=1
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 2: TTS  inference ..."
    mkdir -p ${inference_dir}
    for part in $vc_test_sets; do
        mkdir -p ${inference_dir}/${part}
        echo "inference on set: ${part}"
        data_json="data.JOB.json" # your val set .json file

        utils/run.pl --max-jobs-run 8 JOB=0:$[${ngpu}-1] \
          ${inference_dir}/${part}/inference.JOB.log \
          python3 ../../whisper_infer.py \
            --exp_dir exp/${tag} \
            --rank JOB \
	          --inference_mode 'sampling' \
            --n_samples 1 \
            --seed 888 \
            --rank JOB \
	          --data_json $data_json \
            --generate_target audio \
            --fixed_length False \
            --maxlen_ratio 7 \
            --minlen_ratio 0.5 \
	          --output_dir ${inference_dir}/${part}/JOB \
            $inference_opts
    done
fi