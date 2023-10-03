import json
import argparse
import sys
import os
from utils.task_definition import task_formats

def main(cmd):
    # parse the task first
    possible_tasks = list(task_formats.keys())
    parser = argparse.ArgumentParser(
        description="Build the data json file for data loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, choices=possible_tasks, help="exact task")
    parser.add_argument("--out-json", type=str, help="output json file")
    args, _ = parser.parse_known_args(cmd)

    # add arguments according to the task
    task_format = task_formats[args.task]
    required_files = task_format['keys'] + task_format['features']
    for key in required_files:
        parser.add_argument(f"--{key}", type=str, required=True)
    args = parser.parse_args(cmd)

    # replace the key / feature list by the corresponding key-value dict
    keys = {k: getattr(args, k) for k in task_format['keys']}
    features = {k: getattr(args, k) for k in task_format['features']}

    task_format['keys'] = keys
    task_format['features'] = features
    task_format["task"] = args.task

    # save as a json
    with open(args.out_json, 'wb') as f:
        f.write(
            json.dumps(
                task_format, indent=4, ensure_ascii=False, sort_keys=False
            ).encode("utf_8")
        )

if __name__ == "__main__":
    main(sys.argv[1:])
