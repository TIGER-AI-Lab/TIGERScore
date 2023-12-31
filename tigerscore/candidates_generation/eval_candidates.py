"""
    Eval results will be continuously saved to ../../data/prepared/{dataset_name}/{set_name}/dataset.jsonl
"""
from pathlib import Path
from common.evaluation import (
    overall_eval,
    SUPPORTED_METRICS
)
from common.utils import (
    seed_everything,
    str2bool,
)
import fire
import sys
import os
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_prepared(
    dataset,
    set_name,
    data_dir,
):
    """
    Save evaluated candidates in our format to a single file for each dataset and set.
    Save format:
    {
        "id": str,
        "input": str,
        "output": str,
        "candidates": [
            {
                "decoding_method": str,
                "model": str,
                "text": str,
                "scores": {
                    "bleu": float,
                    "rouge1": float,
                    "rouge2": float,
                    "rougeL": float,
                    "rougeLsum": float,
                    ...
                    }
            },
            ...
        ]
    }


    Args:
        dataset (HuggingFaceDataset): dataset to evaluate
        set_name (str): set to evaluate
        data_dir (str/Path): path to the data directory
    """
    ds_path = Path(data_dir) / dataset / f"{set_name}_data.json"
    save_prepared_path = Path(data_dir) / dataset / \
        f"{set_name}_data_prepared.json"
    assert ds_path.exists(), f"{ds_path} does not exist"
    with open(ds_path) as f:
        ds_data = json.load(f)
    # load candidates
    candidates_dir = Path(data_dir) / dataset / "candidates" / set_name
    decoding_method_dirs = [x for x in candidates_dir.iterdir() if x.is_dir()]
    for decoding_method_dir in decoding_method_dirs:
        decoding_method = decoding_method_dir.name
        # load candidates with eval scores
        candidate_eval_files = [x for x in decoding_method_dir.iterdir(
        ) if x.is_file() and x.suffixes[-2:] == [".eval", ".json"]]
        for candidate_eval_file in candidate_eval_files:
            model_name = candidate_eval_file.stem.split(".")[0]
            with open(candidate_eval_file) as f:
                eval_candidates = json.load(f)
                eval_candidates = {x["id"]: x["candidates"]
                                   for x in eval_candidates}
            assert set(eval_candidates.keys()) == set([x["id"] for x in ds_data]), \
                f"candidate ids do not match for {dataset} {set_name} {decoding_method} {model_name}"
            for example in ds_data:
                example_id = example["id"]
                if "candidates" not in example:
                    example["candidates"] = []

                target_candidates = eval_candidates[example_id]
                target_candidates = sorted(
                    target_candidates, key=lambda x: x["scores"]["bleu"], reverse=True)
                # target_candidate = random.choice(target_candidates)
                # target_candidate = target_candidates[0]
                # example["candidates"].append({
                #     "decoding_method": decoding_method,
                #     "model": model_name,
                #     "text": target_candidate["text"],
                #     "scores": target_candidate["scores"],
                #     }
                # )

                for eval_candidate in eval_candidates[example_id]:
                    example["candidates"].append({
                        "decoding_method": decoding_method,
                        "model": model_name,
                        "text": eval_candidate["text"],
                        "scores": eval_candidate["scores"],
                    })
    print(
        f"Total no. of {set_name} examples in the aggregated dataset: {len(ds_data)}")
    with open(save_prepared_path, "w") as f:
        json.dump(ds_data, f, indent=4, ensure_ascii=False)
    print(f"Saved aggregated {set_name} data to {save_prepared_path}")


def main(data_dir: str = "../../data", dataset: str = "cnndm", _set: str = "test", seed: int = 42, num_workers: int = 1, overwrite: str = "False", save_prepared: str = "True", metrics: str = "rouge,bleu"):
    """_summary_

    Args:
        data_dir (str, optional): Defaults to "../../data".
        dataset (str, optional):  Defaults to "cnndm".
        _set (str, optional): Split of the dataset. Defaults to "test".
        seed (int, optional): Defaults to 42.
        num_workers (int, optional): Attention, some nerual metric in multiprocess are WIP(comet etc). Defaults to 1.
        overwrite (str, optional):  Defaults to "False".
        save_prepared (str, optional): Defaults to "True".
        metrics (str, optional): Defaults to "rouge,bleu".
    """
    overwrite = str2bool(overwrite)
    save_prepared = str2bool(save_prepared)
    metrics = metrics.split(",")
    datasets = dataset.split(",")
    sets = _set.split(",")

    # seed
    seed_everything(seed)

    # prepare metrics
    if 'rouge' in metrics:
        metrics.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        metrics.remove('rouge')
    assert set(metrics).issubset(set(SUPPORTED_METRICS)), \
        "Unsupported metrics: {}".format(set(SUPPORTED_METRICS) - set(metrics))

    for dataset in datasets:
        dataset = dataset.replace(":", "/")

        for set_name in sets:
            print("Evaluating dataset: {} \t set: {}".format(dataset, set_name))
            # get all the decoding method
            candidates_dir = Path(data_dir) / dataset / "candidates" / set_name
            decoding_methods = [
                f.name for f in candidates_dir.iterdir() if f.is_dir()]
            if len(decoding_methods) == 0:
                print("No candidates generated for {}-{}".format(dataset, set_name))
                continue
            for decoding_method in decoding_methods:
                print("Decoding method: {}".format(decoding_method))
                candidate_files = [f for f in (candidates_dir / decoding_method).iterdir(
                ) if f.is_file() and f.suffix == ".json" and ".eval" not in f.suffixes]
                if len(candidate_files) == 0:
                    print(
                        "No candidates generated for {}-{}-{}".format(dataset, set_name, decoding_method))
                    continue
                for candidate_file in candidate_files:
                    print("Model name: {}".format(candidate_file.stem))
                    # load candidates
                    candidate_eval_file = candidate_file.with_suffix(
                        ".eval.json")
                    # candidate_eval_file = candidate_file.with_suffix("._eval.json")
                    if not candidate_eval_file.exists() or overwrite:
                        print("Create a new eval file: {}".format(
                            candidate_eval_file))
                        candidates = json.load(open(candidate_file, 'r'))
                        # create a new eval file if not exists, seperate from the original candidates file
                        json.dump(candidates, open(candidate_eval_file,
                                  'w'), indent=4, ensure_ascii=False)
                    else:
                        print("Load existing eval file: {}".format(
                            candidate_eval_file))
                        candidates = json.load(open(candidate_eval_file, 'r'))
                    # check if the candidates have already been evaluated (sample check)
                    is_evaluated = True
                    for i, sample in enumerate(candidates):
                        _sample_cands = sample['candidates']
                        for j, _sample_cand in enumerate(_sample_cands):
                            if not (set(metrics) <= set(_sample_cand['scores'].keys())):
                                is_evaluated = False
                                break
                        if not is_evaluated:
                            break
                    # Evaluate
                    if not is_evaluated or overwrite:
                        if is_evaluated:
                            print("Overwrite mode: candidates will be re-evaluated")
                        # load targets and pure candidates
                        DS = json.load(
                            open(Path(data_dir) / dataset / f"{set_name}_data.json", 'r'))
                        DS = {x['id']: x for x in DS}
                        pure_candidates = [
                            [x['text'] for x in item['candidates']] for item in candidates]
                        targets = [DS[x['id']]['output'] for x in candidates]
                        # evaluate
                        scores = overall_eval(
                            pure_candidates, targets, metrics, num_workers)
                        # save
                        assert set(scores.keys()) == set(metrics)
                        for metric in metrics:
                            scores_metric = scores[metric]
                            for i, sample in enumerate(candidates):
                                _sample_cands = sample['candidates']
                                for j, _sample_cand in enumerate(_sample_cands):
                                    _sample_cand['scores'][metric] = scores_metric[i][j]
                        json.dump(candidates, open(candidate_eval_file,
                                  'w'), indent=4, ensure_ascii=False)
                        print("Evaluation results saved to {}".format(
                            candidate_eval_file))
                    else:
                        print("Candidates have already been evaluated, skip")
                    # Report the evaluation results
                    for metric in metrics:
                        scores = [[x['scores'][metric]
                                   for x in item['candidates']] for item in candidates]
                        scores = np.array(scores)
                        print("Metric: {}".format(metric))
                        print("Average Min Score: {:.3f}".format(
                            scores.min(axis=1).mean()))
                        print("Average Max Score: {:.3f}".format(
                            scores.max(axis=1).mean()))
                        print("Average Mean Score: {:.3f}".format(
                            scores.mean(axis=1).mean()))
                        print(
                            "Average Default Top-1 Score: {:.3f}".format(scores[:, 0].mean()))
                        print(
                            "Average Default Bottom-1 Score: {:.3f}".format(scores[:, -1].mean()))
        print("Done for dataset: {}".format(dataset))

        if save_prepared:
            for set_name in sets:
                save_prepared(dataset, set_name, data_dir)

    print("Done for all datasets: {}".format(datasets))


if __name__ == "__main__":
    fire.Fire(main)
