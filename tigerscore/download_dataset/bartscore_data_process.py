import os
import sys
import json
import argparse
import pickle
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--rm_old', action='store_true')
    
    args = parser.parse_args()
    data_dir = args.data_dir

    task_dir = Path(data_dir) / args.task
    for data_file in os.listdir(task_dir):
        if not data_file.endswith('.pkl'):
            continue
        print("Data file: ", data_file)
        data_path = task_dir / data_file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print("# of data: ", len(data))
        if isinstance(data, dict):
            print("Data Example: ", data[list(data.keys())[0]])
        elif isinstance(data, list):
            print("Data example: ", data[0])
        with open(data_path.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=4)
        if args.rm_old:
            data_path.unlink()
