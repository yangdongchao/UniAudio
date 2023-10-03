import os
import numpy as np

def compute_eer(fnr, fpr, scores=None):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))

    if scores is not None:
        score_sort = np.sort(scores)
        return fnr[x1] + a * (fnr[x2] - fnr[x1]), score_sort[x1]

    return fnr[x1] + a * (fnr[x2] - fnr[x1])

def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided to
    equalize the counts over score partitions (if there is such partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones((labels.shape), dtype='f8')

    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')

    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    return fnr, fpr

def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det / c_def

def get_id(s):
    s = s.split('SV_')[-1]
    name1 = ''
    new_s = ''
    for i in range(2, len(s)):
        if s[i] != '-':
            name1 = name1 +s[i]
        else:
            new_s = s[i:]
            break
    new_s = new_s.split('_id')[-1]
    name2 = ''
    for i in range(len(new_s)):
        if new_s[i] != '-':
            name2 = name2 + new_s[i]
        else:
            break
    return name1, name2


def compute_metrics(scores_file, label_file, p_target=0.01, c_miss=1, c_fa=1):
    scores = []
    labels = []

    with open(scores_file) as readlines:
        for line in readlines:
            tokens = line.strip().split(' ')
            prob = int(tokens[-1])
            scores.append(bool(prob))
    
    with open(label_file) as readlines:
        for line in readlines:
            tokens = line.strip().split(' ')
            lb = int(tokens[-1])
            labels.append(bool(lb))

    scores = np.hstack(scores)
    labels = np.hstack(labels)
    print('scores ', scores.shape, labels.shape)
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnr, fpr, scores)

    min_dcf = compute_c_norm(fnr,
                             fpr,
                             p_target=p_target,
                             c_miss=c_miss,
                             c_fa=c_fa)
    print("---- {} -----".format(os.path.basename(scores_file)))
    print("EER = {0:.3f}".format(100 * eer))
    print("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.3f}".format(
        p_target, c_miss, c_fa, min_dcf))


scores_file = ''
label_file = ''
compute_metrics(scores_file, label_file)
