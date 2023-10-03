import os
import argparse
import glob
import sys
# please fisrt git clone the audioldm_eval, and then sys.path.append the audioldm_eval
sys.path.append('audioldm_eval')
import torch
from audioldm_eval import EvaluationHelper


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute PESQ measure.")

    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )
    args = parser.parse_args()
    device = torch.device(f"cuda:{0}")
    generation_result_path = args.deg_dir
    target_audio_path = args.ref_dir
    evaluator = EvaluationHelper(16000, device)
    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
    )
    #print(f"VISQOL: {np.mean(visqol)}")
