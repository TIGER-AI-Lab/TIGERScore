from setuptools import setup, find_packages

description = """
TIGERScore, a Trained metric that follows Instruction Guidance to perform Explainable, and Reference-free evaluation over a wide spectrum of text generation tasks. 
Different from other automatic evaluation methods that only provide arcane scores, TIGERScore is guided by the natural language instruction to provide error analysis to pinpoint the mistakes in the generated text.
"""

setup(
    name='tigerscore',
    version='0.0.1',
    description=description,
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://tiger-ai-lab.github.io/TIGERScore/',
    install_requires=[
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'gradio',
        'tiktoken',
        'llama-cpp-python',
        'protobuf',
        'sentencepiece',
        'accelerate'
    ],
)
