## Installation
to get our experiments results, first create a `tigerscore_baseline` environment
```bash
conda create -n tigerscore_baseline python=3.9
conda activate tigerscore_baseline
pip install -r requirements.txt
pip install https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-cp39-cp39-manylinux1_x86_64.whl
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```
