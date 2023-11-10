# The task in slurm connot support long time download,so just download in shell.
from model_utils import build_model, build_tokenizer
import os
from pathlib import Path
import fire


class DownloadModel:
    def __init__(
        self, models: str = None, model_type: str = None, cache_dir: str = None
    ):
        self.models = models
        self.model_type = model_type
        self.cache_dir = (
            cache_dir or Path(os.path.abspath(__file__)).parent.parent.parent / "hf_models"
        )

    def build(self):
        for model in self.models.split(","):
            tokenizer = build_tokenizer(
                model,
                cache_dir=self.cache_dir,
                resume_download=True,
                trust_remote_code=True,
            )
            model = build_model(
                self.model_type,
                model,
                cache_dir=self.cache_dir,
                resume_download=True,
                trust_remote_code=True,
            )


if __name__ == "__main__":
    fire.Fire(DownloadModel)
