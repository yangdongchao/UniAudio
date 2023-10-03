from sklearn import metrics
import sys
import os
import glob
import argparse
from tqdm import tqdm
from scipy.io import wavfile
from pystoi import stoi
import numpy as np
from tools.tokenizer.AudioTagging.audio_tagging_tokenizer import AudioTaggingTokenizer
att = AudioTaggingTokenizer()
def sparse_str_to_array(s): # sparse the str into 
    ans = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] == '1':
            ans[i] = 1
    return ans

def cal_ap_score(gt_file=None, pre_file=None, results=None):
    # we expect input a gt_file: name: value, and its corresponding pre_file, name: value
    if results is not None:
        f_res = open(results, 'r')
        target = []
        clipwise_output = []
        AP = []
        AUC = []
        for line in f_res:
            ans = line.strip().split('class_event')[-1][1:]
            res = ans.split('--')
            if len(res) < 2:
                pre = ''
                gt = res[0]
            else:
                pre = res[0]
                gt = res[1]
            pre = ans.split('--')[0]
            
            pre_arr = att.de_label(pre)
            gt_arr = att.de_label(gt)
            average_precision = metrics.average_precision_score(gt_arr, pre_arr, average=None)
            AP.append(average_precision)
        statistics = {'average_precision': np.mean(AP)} # 
        return statistics
    else:
        target = []
        clipwise_output = []
        f_gt = open(gt_file, 'r')
        for line in f_gt:
            ans = line.strip().split(' ')
            s = ans[-1]
            target.append(sparse_str_to_array(s))
        for line in pre_file:
            ans = line.strip().split(' ')
            s = ans[-1]
            clipwise_output.append(sparse_str_to_array(s))
        average_precision = metrics.average_precision_score(target, clipwise_output, average=None)
        auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        statistics = {'average_precision': average_precision, 'auc': auc}
        return statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute STOI measure")
    parser.add_argument(
        '-r',
        '--ref_dir',
        default=None,
        help="Reference wave folder."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        default=None,
        help="Degraded wave folder."
    )
    parser.add_argument(
        '-a',
        '--results',
        default=None,
        help="point out a results file")
    args = parser.parse_args()
    statistics = cal_ap_score(args.ref_dir, args.deg_dir, args.results)
    print(f"average_precision: {statistics['average_precision']}")
    #print(f"auc: {statistics['auc']}")