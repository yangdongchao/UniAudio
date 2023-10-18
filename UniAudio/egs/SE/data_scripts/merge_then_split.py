# UniAudio Teams
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
from tools.tokenizer.Text2Phone.Text2PhoneTokenizer import Text2PhoneTokenizer
from tools.tokenizer.BPE.SPTokenizer import BPETokenizer
from tools.tokenizer.Semantic.Semantic_tokenizer import SemanticTokenizer
from tools.tokenizer.phone.phone_tokenizer import PhoneTokenizer
from tools.tokenizer.Text_Image.BPETokenizer import BPETokenizer as ClipBPETokenizer

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="the previous .pt save path")
    parser.add_argument("--key-word", type=str, default=None, help="the key word to choose file")
    parser.add_argument("--split-file", type=str, default=None, help="the reference file for split operation")
    return parser

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    data_dict = {}
    
    names = os.listdir(args.input_file)
    for name in names:

    if args.input_file is not None:
        iterator = open(args.input_file)
        for i, line in enumerate(open(args.input_file)):
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
            if i > 0 and i % 1000 == 0:
                logging.info(f"processed {i} examples")
    elif args.tar_file is not None:
        # we use tar as chunk
        iterator = open(args.input_file)
        for i, line in enumerate(open(args.input_file)):
            tar_path = line.strip() # 
            tar_o = tarfile.open(tar_path, mode='r')
            for info in tar_o_1: # get the speech info
                if info.name.split(".")[-1] == args.tar_key_word:
                    cur_f = tar_o_1.extractfile(info)
                    try:
                        cur, _ = librosa.load(BytesIO(pickle.load(cur_f).item()), sr=16000)
                    except:
                        logging.error(f"an error instance: {tar_path} {info.name}")
                        cur = None
                    if cur is None:
                        continue
                    wav = torch.from_numpy(cur).unsqueeze(0) # transfer to (1,len)
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
                    if i > 0 and i % 100 == 0:
                        logging.info(f"processed {i} examples")
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
