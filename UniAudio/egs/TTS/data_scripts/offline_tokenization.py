# Author: # UniAudio Teams

import sys
import torch
import argparse
import logging
import tarfile
import mmap
import pickle
import librosa
from io import BytesIO
from kaldiio import ReadHelper
from tools.tokenizer.soundstream.AudioTokenizer import AudioTokenizer
from tools.tokenizer.soundstream.EncodecTokenizer import EncodecTokenizer
#from tools.tokenizer.Text2Phone.Text2PhoneTokenizer import Text2PhoneTokenizer
from tools.tokenizer.BPE.SPTokenizer import BPETokenizer
from tools.tokenizer.Semantic.Semantic_tokenizer import SemanticTokenizer
from tools.tokenizer.phone.phone_tokenizer import PhoneTokenizer
from tools.tokenizer.Text_Image.BPETokenizer import BPETokenizer as ClipBPETokenizer
from tools.tokenizer.AudioTagging.audio_tagging_tokenizer import AudioTaggingTokenizer 
from tools.tokenizer.CLAP_RVQ.clap_rvq_tokenizer import CLAPRVQTokenizer
def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="text file in the format <exampe_id> <content>")
    parser.add_argument("--tar-file", type=str, default=None, help="we use tar chunk to save audio information")
    parser.add_argument("--tar-key-word", type=str, default=None, help="the key word to find file from tar")
    parser.add_argument("--tar-info", type=str, default=None, help="the file to save tar information")
    parser.add_argument("--wav-scp", type=str, default=None, help="kaldi wav.scp file")
    parser.add_argument("--segments", type=str, default=None, help="kaldi segment file")
    parser.add_argument("--output-file", type=str, help="dict")
    parser.add_argument("--tokenizer", type=str, choices=['audio', 'g2p', 'bpe', 'clipbpe', 'semantic', 'encodec', 'alignment', 'AT', 'clapRVQ'], help="what tokenizer to use")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--batch-size", type=int, default=1, help="for batch tokenization")
    return parser

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    args.rank = (args.rank % max_gpu) #

    if args.tokenizer in ['audio', 'semantic', 'encodec']:
        device = torch.device(f"cuda:{args.rank}")
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    if args.tokenizer in ['g2p', 'bpe', 'alignment', "clipbpe"]:
        assert args.wav_scp is None, f"when using {args.tokenizer}"
        assert args.batch_size == 1, f"{args.tokenizer} doesn't support batch tokenization"

    # GPU tokenizers 
    if args.tokenizer == "audio":
        tokenizer = AudioTokenizer(device=device)
    elif args.tokenizer == "encodec":
        tokenizer = EncodecTokenizer(device=device)
    elif args.tokenizer == "semantic":
        tokenizer = SemanticTokenizer(device=device)
    elif args.tokenizer == 'clapRVQ':
        tokenizer = CLAPRVQTokenizer(device=device)
    # CPU tokenizers
    elif args.tokenizer == "g2p":
        pass
        #tokenizer = Text2PhoneTokenizer()
    elif args.tokenizer == "bpe":
        tokenizer = BPETokenizer()
    elif args.tokenizer == "clipbpe":
        tokenizer = ClipBPETokenizer()
    elif args.tokenizer == "alignment":
        tokenizer = PhoneTokenizer(duplicate=True)
    elif args.tokenizer == "AT":
        tokenizer = AudioTaggingTokenizer()
    else:
        raise NotImplementedError
    tokenizer = tokenizer.to(device)
    logging.info('tokenizer built')
    data_dict = {}
    assert not (args.input_file is not None and args.wav_scp is not None)
    # TODO: support batch inference
    if args.input_file is not None:
        iterator = open(args.input_file)
        s_cnt = 0
        for i, line in enumerate(open(args.input_file)):
            try:
                line = line.strip().split()
                key, value = line[0], " ".join(line[1:])
                value = tokenizer.tokenize(value)
                if value == None:
                    logging.error(f"an error instance: {key} {value}")
                    continue
                if isinstance(value, torch.Tensor):
                    # may not be true for continuous tokenizer
                    # Keep this assertion at this moment.
                    assert value.dim() == 1
                    value = value.cpu()
                data_dict[key] = value
                s_cnt += 1
                if i > 0 and i % 1000 == 0:
                    logging.info(f"processed {s_cnt} examples")
            except:
                logging.error(f"an error instance: {line}")
    elif args.tar_file is not None:
        # we use tar as chunk
        #iterator = open(args.tar_file)
        f_info = open(args.tar_info, 'w')
        for i, line in enumerate(open(args.tar_file,'r')):
            tar_path = line.strip().split(' ')[-1] # 
            try:
                tar_o = tarfile.open(tar_path, mode='r')
                for info in tar_o: # get the speech info
                    # print('tar_path ', tar_path)
                    if info.name.split(".")[-1] == args.tar_key_word:
                        key = ' '.join(info.name.split(".")[:-1]) # 
                        tar_new_name = tar_path.replace('','')
                        tar_new_name = tar_new_name.replace('/', '_')[:-4]
                        key = tar_new_name+'_'+key
                        print('key ', key, info.name)
                        # assert 1==2
                        cur_f = tar_o.extractfile(info)
                        # cur, _ = librosa.load(BytesIO(pickle.load(cur_f).item()), sr=16000)
                        # print('cur ', cur.shape)
                        try:
                            cur = pickle.load(cur_f)
                            #cur, _ = librosa.load(BytesIO(pickle.load(cur_f).item()), sr=16000)
                        except:
                            logging.error(f"an error instance: {tar_path} {info.name}")
                            cur = None
                        if cur is None:
                            continue
                        wav = torch.from_numpy(cur).unsqueeze(0).float() # transfer to (1,len)
                        value = tokenizer.tokenize(wav)
                        if value == None:
                            logging.error(f"an error instance: {key} {value}")
                            continue
                        if isinstance(value, torch.Tensor):
                            # may not be true for continuous tokenizer
                            # Keep this assertion at this moment.
                            assert value.dim() == 1
                            value = value.cpu()
                        data_dict[key] = value
                        f_info.write(key+'\n')
                        if i > 0 and i % 1 == 0:
                            logging.info(f"processed {i} examples")
                tar_o.close()
            except:
                logging.error(f"an error instance: {tar_path}")
    else:
        # kaldiio format
        assert isinstance(tokenizer, AudioTokenizer)
        iterator = ReadHelper('scp:'+args.wav_scp, args.segments)
        count = 0
        for key, (sr, value) in iterator:
            value = torch.from_numpy(value.copy())  / 32768 # [channel, samples]
            value = value.unsqueeze(0)
            value = tokenizer.tokenize(value)
            data_dict[key] = value
            if count > 0 and count % 100 == 0:
                logging.info(f"processed {count} examples")
            count += 1
    torch.save(data_dict, args.output_file)

if __name__ == "__main__":
    main(sys.argv[1:])
