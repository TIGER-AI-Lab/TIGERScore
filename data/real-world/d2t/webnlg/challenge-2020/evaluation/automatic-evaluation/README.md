#  Automatic Evaluation

## Installation

Automatic evaluation scripts are submodules; they have their own GitHub repositories. To install them together with this repository, run

```
git clone --recursive [repo]
```

If the repository is already cloned, run

```
git submodule update --init --recursive
```

## Triples-to-Text

Generation is evaluated with automatic metrics: BLEU, METEOR, chrF++, TER, BERTScore, and BLEURT (for English only).


## Text-to-Triples

Semantic parsing is evaluated with F-score, Precision, and Recall, based on four kinds of matching on an element-level:

- Strict: for each element of the triple, exact match of the candidate string with the reference is required, and the element type (subject, predicate, object) should match with the reference.

- Exact: for each element of the triple, exact match of the candidate string with the reference is required, and the element type (subject, predicate, object) is irrelevant.

- Partial: for each element of the triple, the candidate string should match at least partially with the reference string, and the element type (subject, predicate, object) is irrelevant.

- Type: for each element of the triple, the candidate string should match at least partially with the reference string, and the element type (subject, predicate, object) should match with the reference.

## Leaderboards

Outputs of systems on the development sets can be submitted to [BENG](https://beng.dice-research.org/gerbil/) leaderboards. The outputs are evaluated with official automatic scripts.

The development set is distributed via several folders and files. For leaderboard submission, your outputs should be merged to a single file. The reference data is a concatenation of all files. It is ordered from the 1triples folder to 7triples folder and is a merge of all files by alphabet. The example of the reference file is [here](https://webnlg-challenge.loria.fr/files/reference0-dev.txt).

BENG follows the [FAIR principles](https://www.go-fair.org/fair-principles/) and uses Uniform Resource Identifiers (URI) for maintaining findable links. Each experiment has therefore a unique URI, so anybody can access the experiment later and/or include it in papers.
