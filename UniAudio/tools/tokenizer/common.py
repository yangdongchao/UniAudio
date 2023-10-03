import torch
import random

def clip_by_length(x, length, factor):

    if len(x) <= length:
        return x

    start = random.randint(0, len(x) - length - 1)
    start = start // factor * factor
    x = x[start: start + length]
    return x

def speech_edit_find_time_stamp(x, token_list):
    assert isinstance(x, torch.Tensor)
    x, counts = torch.unique_consecutive(x, return_counts=True)
    x = [token_list[i.item()] for i in x]
    counts = torch.cumsum(counts, dim=0)
    counts = counts.cpu().tolist()

    # Possible Phones obtained from kaldi: 
    # (B)egin, (E)nd, (I)nternal and (S)ingleton 
    # & SIL & SPN_S
    # The phone_table doesn't contain SPN_S so it is replaced by <UNK>
    ans, buf = [], []
    for phone, count in zip(x, counts):
        if phone.endswith('_B') or phone.endswith('_I') or phone.endswith("_E"):
            buf.append((phone, count))
            if phone.endswith("_E"):
                phone_seq = tuple([x[0] for x in buf])
                count = buf[-1][1]
                ans.append((phone_seq, count))
                buf = []
        elif phone == "SIL" or phone.endswith('_S'):
            ans.append((phone, count))
        else:
            ans.append((phone, count)) # usually  SPN_S

    # If too short, mask it all.
    if len(ans) <= 2:
        return (0, ans[-1][1])

    num = random.randint(1, 2) # mask 1-2 words
    word_start = random.randint(0, len(ans) - num)

    if word_start == 0:
        start = 0
    else:
        start = ans[word_start - 1][1]
        
    end = ans[word_start + num - 1][1]

    return (start, end)

def codec_specaug(codec, mask_id):
    """  
    Simply specaug on codec audio input.
    Apply time mask with max-width 5% of the total length; 10 masks
    Apply codec (frequency) mask with only 0 / 1 bin. 1 mask.
    """
    T, D = codec.size()
    max_len = int(T * 0.05)

    for i in range(5):
        start = random.randint(0, T - max_len - 1)
        length = random.randint(0, max_len)
        codec[start: start + length] = mask_id

    if random.random() > 1.0:
        dim = random.randint(0, D - 1)
        codec[:, dim] = mask_id

    return codec.view(-1).contiguous()
    
