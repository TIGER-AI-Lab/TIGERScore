# **TIGERScore**
This repo contains the code, data, and models for "[TIGERScore: Towards Building Explainable Metric for All Text Generation Tasks](https://arxiv.org/abs/2310.00752)"


<div align="center">
 🔥 🔥 🔥 Check out our <a href = "https://tiger-ai-lab.github.io/TIGERScore/">[Project Page]</a> for more results and analysis!
</div>

<br>
<div align="center">
  <img src="github_overview.png" width="80%" title="Introduction Figure">
</div>

## 🔥News

- [12/2] TIGERScore now support running with llama.cpp, check [Quantization Support Cpu](#quantization-support-cpu) for details

## **Table of Contents**

- [📌 Introduction](#introduction)
- [🤗 Datasets and Models](#datasets-and-models)
- [⚙️ Installation](#installation)
- [🛠️ Usage](#usage)
- [📜 License](#license)
- [📖 Citation](#citation)




## **Introduction**
We present 🐯 TIGERScore, a **T**rained metric that follows **I**nstruction **G**uidance to perform **E**xplainable, and **R**eference-free evaluation over a wide spectrum of text generation tasks. 

Existing automatic metrics either are lagging and suffer from issues like 1) **Dependency on references**, 2) **Limited to specific domains**, 3) **Lack of attribution**. Contrary to them, TIGERScore is designed to be driven by natural language instruction and provide detailed error analysis to pinpoint the mistakes in the generated text.

Specifically, TIGERScore takes an instruction, an associated input context along with a hypothesis output that might contain errors. Then, TIGERScore will evaluate this hypothesis output and list several errors, each consisting of the error location, aspect, explanation and penalty scores (score reduced, starting from 0). The sum of the reduced scores is taken as the overall rating of this output. 

Experiments show that TIGERScore surpass existing baseline metrics in correlation with human ratings on all 6 held-in tasks and 1 held-out task, achiving the highest overall performance. We hope the emergence of TIGERScore can promote the research in the LLM community as a powerful, interpretable, and easy-to-use metric.

## Datasets and Models

| Datasets |
| ----- |
| [📏 MetricInstruct](https://huggingface.co/datasets/TIGER-Lab/MetricInstruct) |

| Models 🐯                                           	 | 
|---------------------------------------------------------------	 |
|  🦙 [TIGERScore-7B](https://huggingface.co/TIGER-Lab/TIGERScore-7B)   	| 
|  🦙 [TIGERScore-13B](https://huggingface.co/TIGER-Lab/TIGERScore-13B) 	| 
|  🦙 [TIGERScore-7B-GGUF](https://huggingface.co/TIGER-Lab/TIGERScore-7B-GGUF)   	| 
|  🦙 [TIGERScore-13B-GGUF](https://huggingface.co/TIGER-Lab/TIGERScore-13B-GGUF) 	| 
|  <img src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi.svg" style="height: 1em; vertical-align: middle;" title="Yi"> [TIGERScore-Yi-6B](https://huggingface.co/TIGER-Lab/TIGERScore-Yi-6B) |



## Installation

To directly use tigerscore pipeline, you first need to install it as python package. 
```bash
# create enronments
conda create -n tigerscore python=3.9
conda activate tigerscore
# install torch cuda toolkits
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
# install tigerscore python package
pip install git+https://github.com/TIGER-AI-Lab/TIGERScore.git
```
Please do check if your `torch.cuda.is_available()` is `True` for your local machine.

Besides, to use TIGERScore with vllm detailed [here](#vllm-support-recommended), you need to mannually install vllm following [vllm document](https://docs.vllm.ai/en/latest/getting_started/installation.html). For example, if your CUDA is 12.1, you can simply install with `pip install vllm`

## Usage

### Basic Usage
After installation, you are good to score the text generations with the following exmaple python code (see in [`tigerscore_example_usage.ipynb`](./tigerscore_example_usage.ipynb) for more use cases) :
```python
# gpu device setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set up scorer
from tigerscore import TIGERScorer
scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B") # on GPU
# scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True) # 4 bit quantization on GPU
# scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True) # VLLM on GPU
# scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True) # 4 bit quantization on CPU

# example  
instruction = "Write an apology letter."
input_context = "Reason: You canceled a plan at the last minute due to illness."
hypo_output = "Hey [Recipient],\n\nI'm really sorry for ditching our plan. I suddenly got an opportunity for a vacation so I took it. I know this might have messed up your plans and I regret that.\n\nDespite being under the weather, I would rather go for an adventure. I hope you can understand my perspective and I hope this incident doesn't change anything between us.\n\nWe can reschedule our plan for another time. Sorry again for the trouble.\n\nPeace out,\n[Your Name]\n\n---"
results = scorer.score([instruction], [hypo_output], [input_context])
print(results)
``` 
The results is a list of dicts consisting of structured error analysis.
```json
[
    {
        "num_errors": 2,
        "score": -7.0,
        "errors": {
            "error_0": {
                "location": " \"I suddenly got an opportunity for a vacation so I took it.\"",
                "aspect": " Misunderstanding context",
                "explanation": " The error lies in the context of the reason for cancelling the plan. The original reason was due to illness, but in the incorrect output, it is stated that the cancellation was due to a vacation opportunity, which is a misunderstanding of the context. The correction would be to stick to the original reason for cancelling.",
                "severity": "Major",
                "score_reduction": "5.0"
            },
            "error_1": {
                "location": " \"I hope you can understand my perspective and I hope this incident doesn't change anything between us.\"",
                "aspect": " Inappropriate tone",
                "explanation": " The tone of this sentence is too casual and lacks regret or apology. It's important to maintain a formal and regretful tone in an apology letter. The sentence could be corrected to something like \"I hope you can find it in your heart to forgive me and let this incident not strain our relationship.\"",
                "severity": "Minor",
                "score_reduction": "2.0"
            }
        },
        "raw_output": " The model-generated output contains 2 errors, with a total score reduction of 7.0.\nError location 1: ..."
    }
]
```

### VLLM Support (**Recommended**)
<details>
  <summary>use_vllm=True</summary>
    ```python
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True) # VLLM on GPU
    ```
    TIGERScore supports VLLM fast inference. On a single A6000 (48GB) GPU, it only takes **0.2s - 0.3s** for TIGERScore-13b to score each instance.
</details>

### Quantization Support (GPU)
<details>
  <summary>quantized=True</summary>
    ```python
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True) # 4 bit quantization on GPU
    ```
    By setting the initialization parameter `quanitzed=True`, the model is set to be load in 4-bit version with hugging face `load_in_4bit=True` option. 

    Please note that though using quantization would decrease the memory requirement by a large margin. You can run TIGERScore on about a 20+GB memory GPU. However, the inference speed might be slower than using the original bfloat16 version. It depends on you to make an trade-off.
</details>

### Quantization Support (CPU)
<details>
  <summary>use_llamacpp=True</summary>
    ```python
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True)
    ```
    We also provide the Llamacpp version of TIGERScore-7B/13B. By using the GGUF version we provided, you can run TIGERScore on pure CPU devices. It generally takes **20s** for TIGERScore-13b to score each instance.
</details>

## Data Preparation

### Propmting template
folder [`xgptscore`](./tigerscore/xgptscore/) contains all the templates that we used to query ChatGPT or GPT-4 to get the identified errors in the hypothesis output for different tasks that TIGERScore involved. We call these API query methods as XGPTScore for a e**X**planainable **Scoring** method by querying **GPT** Models.

The overall pipeline of XGPTScore is:

1. We define a query template that askes GPT Models to idnetify errors in the hypothesis output based on the task instruction, source text and reference text.
2. We mannual construct various evaluation aspects to focus on for different tasks. ([`./constants.py`](./tigerscore/xgptscore/constants.py))
3. Then, by applying the templates and also specifiy the aspects to focus on in the template, GPT Models are required to return the identified errors in a predefined format (like json format).

Check [`xgptscore/README.md`](./tigerscore/xgptscore/README.md) for more details. And how to use our query template with a single function `xgptscore()`

### Dataset Components
MetricInstruct consists of data from 2 sampling channels, **real-world channel** and **synthetic channel**. 
- The real-world channel data is generated by script [`generate_distill_data.sh`](./tigerscore/eval_scripts/generate_distill_data.sh).
- The synthetic channel data is generated by script [`generate_synthesis_distill_data.sh`](./tigerscore/eval_scripts/generate_synthesis_distill_data.sh).
The overall purpose of 2 channel data collection is to make sure we cover as many as error types in the training data so that our model generalize better.

After getting these data, we do a series heuristics to filter our bad data and augment data:
1. Drop item that is too long, too short, bad format, etc (pattern matching)
2. Propmt GPT-4 to drop item with unreasonable error analysis contents ([`check_data.sh`](./tigerscore/eval_scripts/check_data.sh))
3. Our evaluation asepcts might be limited because they are mannually defined and fixed. Therefore, we propose to generate high-quality outputs with free-form error asepcts using [`generate_inst_synthetic_data.sh`](./tigerscore/eval_scripts/generate_inst_synthetic_data.sh) as a supplement to the synthetic channel. 

### 📏 MetricInstruct 

You can load our preprocessed data used to finetune TIGERScore-V1 from hugging face 🤗 directly:
```python
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/MetricInstruct")
```

## **Training Scripts**

We provide our training and testing scripts in folder [`finetune`](./tigerscore/finetune/), where we use🧮 
- [`finetune_llama.sh`](./tigerscore/finetune/finetune_llama.sh) to finetine the model.
- [`format_distill_data.sh`](./tigerscore/finetune/format_distill_data.sh) to transform the data into the format for finetuning, that is, a sinlge instruction and input context with an output.
- [`test_llama_vllm.sh`](./tigerscore/finetune/test_llama_vllm.sh) to test and compute the correlation as the performance of our finetuned model. 
Please check these scripts to know more details of our training and testing process.

## **Citation**

Please cite our paper if you fine our data, model or code useful. 

```
@article{jiang2023TIGERScore,
  title={TIGERScore: Towards Building Explainable Metric for All Text Generation Tasks},
  author={Dongfu Jiang, Yishan Li, Ge Zhang, Wenhao Huang, Bill Yuchen Lin, Wenhu Chen},
  journal={arXiv preprint arXiv:2310.00752},
  year={2023}
}
```
