# The task in slurm connot support long time download,so just download in shell.
from model_utils import build_model, build_tokenizer
import os
from pathlib import Path
import fire


def main( models: str = None, model_type: str = None, cache_dir: str = None):
        models = models
        model_type = model_type
        cache_dir = (
            cache_dir or Path(os.path.abspath(__file__)).parent.parent.parent / "hf_models"
        )
        for model in models.split(","):
            tokenizer = build_tokenizer(
                model,
                cache_dir=cache_dir,
                resume_download=True,
                trust_remote_code=True,
            )
            model = build_model(
                model_type,
                model,
                cache_dir=cache_dir,
                resume_download=True,
                trust_remote_code=True,
            )


if __name__ == "__main__":
    fire.Fire(main)