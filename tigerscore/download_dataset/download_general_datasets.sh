#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --job-name=downloading_general_datasets
#SBATCH --output ../../jobs/%j.out
#SBATCH --nodelist=ink-gary
#SBATCH -n 1

python download_general_datasets.py --task "mathQA" --overwrite False
python download_general_datasets.py --task "summarization" --overwrite False
python download_general_datasets.py --task "translation" --overwrite False
python download_general_datasets.py --task "data2text" --overwrite False
python download_general_datasets.py --task "long-form QA" --overwrite False
python download_general_datasets.py --task "instruction-following" --overwrite False
# python download_general_datasets.py --task "story_generation"
# python download_general_datasets.py --task "image_captioning"
python download_general_datasets.py --task "code"