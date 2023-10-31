# WebNLG Challenge 2020: Translation and Baseline Generation for Text-to-RDF

The translation script for the Russion baseline (for the Russian Text-to-Triples task), and the triple extraction script for the [WebNLG Challenge 2020](https://webnlg-challenge.loria.fr/challenge_2020/).  

## Translation Script (Translate_Baseline_Text2rdf.py)
This script will translate the Russian test set to English, which is necessary for the subsequent Triple Extraction task, as the triple extraction tool only works for English. Translation is done using [DeepL](https://www.deepl.com/translator/).

This script needs the following libraries:

- [Pyperclip](https://pypi.org/project/pyperclip/)
- [Selenium](https://pypi.org/project/selenium/)
- [Regex](https://pypi.org/project/regex/)

The command to use the script is: ```python3 Translate_Baseline_Text2rdf.py <input filename> <output filename>```.
Make sure that the input file is in the same folder as the script.

## Baseline Script (SVO_Extraction.py)
This script will extract SVO triples from an input text (the English test set, or the translated Russian test set) using `stanfordcorenlp`, and produce XML output as an [evaluation script](https://github.com/WebNLG/Evaluation/tree/main/automatic-evaluation/text-to-triples) compliant XML file. The script is based on an [SVO Extraction script](https://github.com/junhyeok-kim/TripletEmbeddingModel) by Junhyeok Kim.

To run this script, you need the following libraries:

- [Regex](https://pypi.org/project/regex/)
- [Numpy](https://pypi.org/project/numpy/)
- [StanfordCoreNLP](https://github.com/Lynten/stanford-corenlp)
- [LXML](https://pypi.org/project/lxml/)

The command to use the script is: ```python3 SVO_Extraction.py <input filename> <output filename>```.
Make sure that the input file and the unzipped [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) folder is in the same folder as the script.
