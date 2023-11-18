# %%
import json
import numpy as np
import fire
from scipy.stats import mode

def main(input_file_path, output_file_path):

    new_data = []
    
    # roc_input_file_path = input_file_path + "mans_roc.json"
    # with open(roc_input_file_path, 'r') as f:
    #     data = json.load(f)

    # for key, value in data.items():
    #     cands = []
    #     for _k, _v in value["gen"].items():
    #         cands.append({
    #             "model": _k,
    #             "decoding_method": "greedy",
    #             "text": _v["text"],
    #             # "scores": {"human": int(mode(_v["score"]).mode)},
    #             "scores": {"human": np.mean(_v["score"])},
    #         })
    #     new_data.append({
    #         "id": key,
    #         "instruction": "Generate a reasonable ending for the following story.",
    #         "input": value["prompt"],
    #         "output": value["gold_response"],
    #         "candidates": cands,
    #         "data_source": "roc",
    #     })
    
    wp_input_file_path = input_file_path + "mans_wp.json"
    with open(wp_input_file_path, 'r') as f:
        data = json.load(f)
    for key, value in data.items():
        cands = []
        for _k, _v in value["gen"].items():
            cands.append({
                "model": _k,
                "decoding_method": "greedy",
                "text": _v["text"],
                "scores": {"human": np.mean(_v["score"])},
            })
        new_data.append({
            "id": key,
            "instruction": "Generate a reasonable ending for the following story.",
            "input": value["prompt"],
            "output": value["gold_response"],
            "candidates": cands,
            "data_source": "wp",
        })

    with open(output_file_path, 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
