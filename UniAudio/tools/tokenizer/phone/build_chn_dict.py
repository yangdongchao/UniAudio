import sys

ph_dict = {}
align_file = sys.argv[1]

for i, line in enumerate(open(align_file)):
    line = line.strip().split()[1:]
    for ph in line:
        if ph not in ph_dict:
            ph_dict[ph] = None

    if i % 100000 == 0:
        print(f'processed {i} lines {len(ph_dict)}')

for i, k in enumerate(ph_dict.keys()):
    print(f"{k} {i}")
