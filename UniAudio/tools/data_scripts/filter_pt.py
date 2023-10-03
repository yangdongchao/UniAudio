import torch
import sys

ref_file = sys.argv[1]
out_pt_file = sys.argv[2]
in_pt_files = sys.argv[3:]

ref_dict = open(ref_file).readlines()
ref_dict = {line.strip().split()[0]: None for line in ref_dict}

pt = {}
for f in in_pt_files:
    this_dict = torch.load(f)
    this_dict = {k: v for k, v in this_dict.items() if k in ref_dict}
    pt.update(this_dict)

torch.save(pt, out_pt_file)
