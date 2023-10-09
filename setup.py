from setuptools import setup, find_packages

description = """
TIGERScore, a Trained metric that follows Instruction Guidance to perform Explainable, and Reference-free evaluation over a wide spectrum of text generation tasks. 
Different from other automatic evaluation methods that only provide arcane scores, TIGERScore is guided by the natural language instruction to provide error analysis to pinpoint the mistakes in the generated text.
"""

# transformers==4.33.2
# datasets==2.10.0
# torch
# accelerate
# wget
# pycocoevalcap
# spacy
# evaluate
# prettytable
# fairscale
# bert_score
# gdcm 
# pydicom
# sacremoses
# apache_beam
# deepspeed
# bitsandbytes
# openai
# nltk
# scipy
# json5
# peft
# fire
# gradio
# sentencepiece
# tiktoken
# dacite
# wandb
# rouge_score
# bs4
# py7zr
# sacrebleu
# gdown
# git+https://github.com/google-research/mt-metrics-eval.git

setup(
    name='tigerscore',
    version='0.0.1',
    description=description,
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://tiger-ai-lab.github.io/TIGERScore/',
    install_requires=[
        'transformers>=4.33.2',
        'datasets>=2.10.0',
        'torch',
        'accelerate',
        'wget',
        'pycocoevalcap',
        'spacy',
        'evaluate',
        'prettytable',
        'fairscale',
        'bert_score',
        'gdcm',
        'pydicom',
        'sacremoses',
        'apache_beam',
        'deepspeed',
        'bitsandbytes',
        'openai',
        'nltk',
        'scipy',
        'json5',
        'peft',
        'fire',
        'gradio',
        'sentencepiece',
        'tiktoken',
        'dacite',
        'wandb',
        'rouge_score',
        'bs4',
        'py7zr',
        'sacrebleu',
        'gdown',
        'accelerate',
        'bitsandbytes',
        'mt-metrics-eval @ git+https://github.com/google-research/mt-metrics-eval.git',
    ],
)
