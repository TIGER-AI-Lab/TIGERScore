# **TIGERScore**
This repo contains the code, data, and models for "[TIGERScore: Towards Building Explainable Metric for All Text Generation Tasks](https://arxiv.org/abs/2310.00752)"


<div align="center">
 🔥 🔥 🔥 Check out our <a href = "https://tiger-ai-lab.github.io/TIGERScore/">[Project Page]</a> for more results and analysis!
</div>

<br>
<div align="center">
  <img src="github_overview.png" width="80%" title="Introduction Figure">
</div>

### Datasets and Models
Our dataset and models are all available at Huggingface.

🤗 [MetricInstruct Dataset](https://huggingface.co/datasets/TIGER-Lab/MetricInstruct)

|     	| Base Model: Llama-2                                           	 | 
|-----	|---------------------------------------------------------------	 |
| 7B  	|  [TIGERScore-7B-V1.0](https://huggingface.co/TIGER-Lab/TIGERScore-7B-V1.0)   	| 
| 13B 	|  [TIGERScore-13B-V1.0](https://huggingface.co/TIGER-Lab/TIGERScore-7B-V1.0) 	| 



## **Table of Contents**

- [📌 Introduction](#introduction)
- [⚙️ Installation](#installation)
- [🛠️ Training and Inference](#training-and-inference)
- [📜 License](#license)
- [📖 Citation](#citation)

## **Introduction**
We present 🐯 TIGERScore, a **T**rained metric that follows **I**nstruction **G**uidance to perform **E**xplainable, and **R**eference-free evaluation over a wide spectrum of text generation tasks. Different from other automatic evaluation methods that only provide arcane scores, TIGERScore is guided by the natural language instruction to provide error analysis to pinpoint the mistakes in the generated text.
## **TIGERScore Usage**

### Installation

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
### GPU running
After installation, you are good to score the text generations with the following exmaple python code (see in [`tigerscore_example_usage.ipynb`](./tigerscore_example_usage.ipynb) for more use cases) :
```python
# gpu device setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set up scorer
from tigerscore import TIGERScorer
scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-13B-V1.0"
, quantized=False) # set quantized=False to use bfloat16 version on gpu deviced
# scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-13B-V1.0", quantized=True) # set quantized=True to use 4-bit version on either gpu or cpu.
# load the dataset
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/MetricInstruct")
num_few_examples = 10
tasks = dataset["train_mix"]['task'][0:num_few_examples]
insts = dataset["train_mix"]['instruction'][0:num_few_examples]
input_contexts = dataset["train_mix"]['input_context'][0:num_few_examples]
hypo_output = dataset["train_mix"]['hypo_output'][0:num_few_examples]
# scoring
results = scorer.score(tasks, insts, input_contexts, hypo_output, batch_size=8)
scores = [result["score"] for result in results] 
print(scores) # List of float scores
print(results[0]) # associated explanation texts
``` 
The device used to automatically set through huggingface `device_map="auto"` option.

### CPU running and Quantization
By setting the initialization parameter `quanitzed=True`, the model is set to be load in 4-bit version with hugging face `load_in_4bit=True` option. 

To run it on cpu, please first set `CUDA_VISIBLE_DEVICES=""` to disable GPU. Then set `quanitzed=True` when initializing the scorer. Then with the same code above. You are good to go with tigerscore.

Please note that though using quantization would decrease the memory requirement by a large margin, the inference speed might be slower than using the original bfloat16 version. It depends on you to make an trade-off.

## Data Preparation

### GPTModels template
folder [`xgptscore`](./tigerscore/xgptscore/) contains all the templates that we used to query ChatGPT or GPT-4 to get the identified errors in the hypothesis output for different tasks that TIGERScore involved. We call these API query methods as XGPTScore for a e**X**planainable **Scoring** method by querying **GPT** Models.

The overall pipeline of XGPTScore is:

1. We define a query template that askes GPT Models to idnetify errors in the hypothesis output based on the task instruction, source text and reference text.
2. We mannual construct various evaluation aspects to focus on for different tasks.
3. Then, by applying the templates and also specifiy the aspects to focus on in the template, GPT Models are required to return the identified errors in a predefined format (like json format).

Check [`xgptscore/README.md`](./tigerscore/xgptscore/README.md) for more details. And how to use our query template with a single function `xgptscore()`


### 📏 MetricInstruct 

You can load our preprocessed data used to finetune TIGERScore-V1 from hugging face 🤗 directly:
```python
from datasets import load_dataset

dataset = load_dataset("TIGER-Lab/MetricInstruct")
```

MetricInstruct consists of data from 2 sampling channels, **real-world channel** and **synthetic channel**. 
- The real-world channel data is generated by script [generate_distill_data.sh](./tigerscore/eval_scripts/generate_distill_data.sh).
- The synthetic channel data is generated by script [generate_synthesis_distill_data.sh](./tigerscore/eval_scripts/generate_synthesis_distill_data.sh).
The overall purpose of 2 channel data collection is to make sure we cover as many as error types in the training data so that our model generalize better.

## **Training Scripts**

We provide our training and testing scripts in folder [finetune](./tigerscore/finetune/), where we use🧮 
- [finetune_llama.sh](./tigerscore/finetune/finetune_llama.sh) to finetine the model.
- [format_distill_data.sh](./tigerscore/finetune/format_distill_data.sh) to transform the data into the format for finetuning, that is, a sinlge instruction and input context with an output.
- [test_llama.sh](./tigerscore/finetune/test_llama.sh) to test and compute the correlation as the performance of our finetuned model. 
Please check these scripts to know more details of our training and testing process.

## **Citation**

Please cite our paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@article{jiang2023TIGERScore,
  title={TIGERScore: Towards Building Explainable Metric for All Text Generation Tasks},
  author={Dongfu Jiang, Yishan Li, Ge Zhang, Wenhao Huang, Bill Yuchen Lin, Wenhu Chen},
  journal={arXiv preprint arXiv:2310.00752},
  year={2023}
}
```
