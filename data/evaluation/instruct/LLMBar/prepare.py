import json
import sys,os

input_file = "./dataset.json"
datas = []
def get_data(input_file):
    with open(input_file) as f:
        data = json.load(f)

        for d in data:
            datas.append({
                "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                "input": d["input"],
                "output": None,
                "candidates": [
                    {
                        "text": d["output_1"],
                        "source": "1",
                        "scores": {
                            "label": 1 if d["label"] == 1 else 0,
                        }
                    },
                    {
                        "text": d["output_2"],
                        "source": "2",
                        "scores": {
                            "label": 1 if d["label"] == 2 else 0,
                        }
                    }
                ]
            })


get_data(input_file)
# get_data(input_file3)
# get_data(input_file4)
with open("test.json", "w") as f:
    json.dump(datas, f, indent=4, ensure_ascii=False)
