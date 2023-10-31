# WebNLG+ Challenge 2020

The repository with baselines, submissions, evaluation scripts and results from the 2nd version of the [WebNLG+ Challenge 2020](https://webnlg-challenge.loria.fr/challenge_2020/).

## Challenge Overview

* Webpage: <https://webnlg-challenge.loria.fr/challenge_2020/>
* Training, validation and test data: [WebNLG dataset version 3.0](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0)
* Automatic evaluation scripts:
	* [RDF-to-text](https://github.com/WebNLG/GenerationEval)
	* [text-to-RDF](https://github.com/WebNLG/WebNLG-Text-to-triples)
* BENG leaderboard: <https://beng.dice-research.org/gerbil/>
* Papers:
	* Challenge report: [ACL Anthology](https://www.aclweb.org/anthology/2020.webnlg-1.7/), presentation [slides](https://webnlg-challenge.loria.fr/files/WebNLG-2020-Presentation.pdf) at INLG'2020
	* BENG leaderboard paper: [ACL Anthology](https://www.aclweb.org/anthology/2020.webnlg-1.3/)
	* [System descriptions](./submissions#participant-overview)
	* WebNLG+ 2020 workshop proceedings: [ACL Anthology](https://www.aclweb.org/anthology/volumes/2020.webnlg-1/)
	* [Research papers](https://webnlg-challenge.loria.fr/research/) that use the WebNLG dataset

## Repository Structure

* `evaluation/human-evaluation/results` contains human evaluation results for the RDF-to-text task
* `submissions` contains team submissions to the challenge (.txt and .xml files)
* `evaluation/human-evaluation/eval-scripts` contains scripts to calculate human scores, IAA, and statistical significance testing


```bash
.
├── baselines
│   ├── rdf2text
│   └── text2rdf
├── evaluation
│   ├── automatic-evaluation
│   │   ├── text-to-triples
│   │   └── triples-to-text
│   ├── human-evaluation
│   │   ├── amt-scripts
│   │   │   ├── data
│   │   │   │   └── en
│   │   │   │       ├── inputs
│   │   │   │       └── tables
│   │   │   ├── eval_metadata
│   │   │   ├── hit_utils
│   │   │   └── scripts
│   │   │       └── corpus-reader
│   │   ├── eval-scripts
│   │   │   ├── en
│   │   │   └── ru
│   │   └── results
│   │       ├── en
│   │       │   ├── Amazon_AI_(Shanghai)
│   │       │   ├── Baseline-FORGE2017
│   │       │   ├── Baseline-FORGE2020
│   │       │   ├── bt5
│   │       │   ├── cuni-ufal
│   │       │   ├── CycleGT
│   │       │   ├── DANGNT-SGU
│   │       │   ├── FBConvAI
│   │       │   ├── Huawei_Noahs_Ark_Lab
│   │       │   ├── NILC
│   │       │   ├── NUIG-DSI
│   │       │   ├── ORANGE-NLG
│   │       │   ├── OSU_Neural_NLG
│   │       │   ├── RALI
│   │       │   ├── TGen
│   │       │   ├── UPC-POE
│   │       │   └── WebNLG-2020-reference
│   │       └── ru
│   │           ├── Baseline-FORGE2020
│   │           ├── bt5
│   │           ├── cuni-ufal
│   │           ├── FBConvAI
│   │           ├── Huawei_Noahs_Ark_Lab
│   │           ├── med
│   │           ├── OSU_Neural_NLG
│   │           └── WebNLG-2020-reference
│   └── references
└── submissions
    ├── rdf2text
    │   ├── en
    │   │   ├── Amazon_AI_(Shanghai)
    │   │   ├── Baseline-FORGE2017
    │   │   ├── Baseline-FORGE2020
    │   │   ├── bt5
    │   │   ├── cuni-ufal
    │   │   ├── CycleGT
    │   │   ├── DANGNT-SGU
    │   │   ├── FBConvAI
    │   │   ├── Huawei_Noahs_Ark_Lab
    │   │   ├── NILC
    │   │   ├── NUIG-DSI
    │   │   ├── ORANGE-NLG
    │   │   ├── OSU_Neural_NLG
    │   │   ├── RALI
    │   │   ├── TGen
    │   │   └── UPC-POE
    │   └── ru
    │       ├── Baseline-FORGE2020
    │       ├── bt5
    │       ├── cuni-ufal
    │       ├── FBConvAI
    │       ├── Huawei_Noahs_Ark_Lab
    │       ├── med
    │       └── OSU_Neural_NLG
    └── text2rdf
        ├── en
        │   ├── Amazon_AI_(Shanghai)
        │   ├── Baseline
        │   ├── bt5
        │   └── CycleGT
        └── ru
            ├── Baseline
            └── bt5
```

## Installation

This repository contains submodules. To install them together with the repository, run

```
git clone --recursive [repo]
```

If the repository is already cloned, run

```
git submodule update --init --recursive
```

## Citation
If you use the scripts and data provided by the challenge, please cite the [challenge report](https://www.aclweb.org/anthology/2020.webnlg-1.7/).

```
@inproceedings{webnlg2020report,
    title = "The 2020 Bilingual, Bi-Directional {W}eb{NLG}+ Shared Task: Overview and Evaluation Results ({W}eb{NLG}+ 2020)",
    author = "Castro Ferreira, Thiago  and
      Gardent, Claire  and
      Ilinykh, Nikolai  and
      van der Lee, Chris  and
      Mille, Simon  and
      Moussallem, Diego  and
      Shimorina, Anastasia",
    booktitle = "Proceedings of the 3rd International Workshop on Natural Language Generation from the Semantic Web (WebNLG+)",
    month = "12",
    year = "2020",
    address = "Dublin, Ireland (Virtual)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.webnlg-1.7",
    pages = "55--76"
}
```

## Note
This repository is still under development, so some parts may be missing. We thank you for your understanding.
