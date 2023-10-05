import sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="output a file this specifies all peer utterances of each utts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--utt2spk", type=str, help="original spk2utt file")
    parser.add_argument("--peer-utts", type=str, help="revised spk2utt file")
    parser.add_argument("--subset-list", type=str, help="list of utt subset")
    return parser

def main(args):
    args = get_parser().parse_args(args)

    # subset utts
    subset_utts = open(args.subset_list).readlines()
    subset_utts = [line.strip().split()[0] for line in subset_utts]

    # utt2spk
    utt2spk = open(args.utt2spk).readlines()
    utt2spk = [line.strip().split() for line in utt2spk]
    utt2spk = {line[0]: line[1] for line in utt2spk}

    # utt2spk
    spk2utt = {}
    for utt, spk in utt2spk.items():
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    writer = open(args.peer_utts, 'w') 
    for utt in utt2spk.keys():
        spk = utt2spk[utt]
        peer_utts = spk2utt[spk]

        if len(peer_utts) <= 1:
            print(f'warning: utt {utt} has no peer speech except itself')

        out_str = " ".join([utt] + peer_utts)
        writer.write(out_str + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])
