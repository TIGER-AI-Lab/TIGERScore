# The task in slurm connot support long time download,so just download in shell.
from model_utils import (
    build_model,
    build_tokenizer
)
import argparse
import os
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type = str, default = None)
    parser.add_argument('--model_type', type = str, default = None)
    parser.add_argument('--cache_dir', type = str, default = None)
    args = parser.parse_args()
    if args.cache_dir is None:
        args.cache_dir = Path(os.path.abspath(__file__)).parent.parent.parent / "hf_models"
    
    for model in args.models.split(','):
        tokenizer = build_tokenizer(model,cache_dir=args.cache_dir,resume_download=True,trust_remote_code=True)
        model = build_model(args.model_type,model,cache_dir=args.cache_dir,resume_download=True,trust_remote_code=True)
        