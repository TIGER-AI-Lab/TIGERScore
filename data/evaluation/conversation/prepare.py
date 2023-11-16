import json
import sys
import os

input_file1 = "./test_freq.json"
input_file2 = "./test_rare.json"
# input_file3 = "./valid_freq.json"
# input_file4 = "./valid_rare.json"

score_func = {
    "Poor": 1,
    "Not Good": 2,
    "Passable": 3,
    "Good": 4,
    "Excellent": 5
}
datas = []


def get_data(input_file):
    with open(input_file) as f:
        data = json.load(f)

        for _, d in data.items():

            _input = d["content"][0]["agent"].capitalize(
            ) + ": " + d["content"][0]["message"] + "\n"
            _output = ""
            if d["content"][0]["turn_rating"] == "":
                continue
            for _, c in enumerate(d["content"][1:]):
                if c["turn_rating"] == "":
                    break
                _output = c["agent"].capitalize() + ": " + c["message"] + "\n"
                datas.append({
                    "id": d["article_url"],
                    "instruction": "Reply to the conversation.",
                    "input": _input,
                    "output": None,
                    "candidates": [
                        {
                            "text": _output,
                            "source": "unknown",
                            "scores": {
                                "turn_rating": score_func[c["turn_rating"]]
                            }
                        }
                    ]
                })
                _input += _output


get_data(input_file1)
get_data(input_file2)
# get_data(input_file3)
# get_data(input_file4)
with open("test.json", "w") as f:
    json.dump(datas, f, indent=4, ensure_ascii=False)
