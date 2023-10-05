import sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="Revise the spk2utt file: it only contans a subset of the utts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in-spk2utt", type=str, help="original spk2utt file")
    parser.add_argument("--out-spk2utt", type=str, help="revised spk2utt file")
    parser.add_argument("--subset-list", type=str, help="list of utt subset")
    return parser

def main(args):
    args = get_parser().parse_args(args)

    utts = open(args.subset_list).readlines()
    utts = [line.strip().split()[0] for line in utts]
    utts = {x: None for x in utts}

    writer = open(args.out_spk2utt, 'w') 
    for line in open(args.in_spk2utt):
        line = line.strip().split()
        spk_id, spk_utts = line[0], line[1:]
        spk_utts = [utt for utt in spk_utts if utt in utts]

        out_str = " ".join([spk_id] + spk_utts)
        writer.write(out_str + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])
