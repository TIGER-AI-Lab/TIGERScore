from datasets import load_dataset

dataset = load_dataset("lmsys/mt_bench_human_judgments")
data = [x for x in dataset['human']]

from collections import Counter
Counter([len(x['conversation_a']) for x in data])

import json
new_data = []
inst = "Finish the following coversation by filling in <Response 1> and <Response 2> in the blank."
inputs = [
    "USER: " + d['conversation_a'][0]['content'] + 
    "\nAssistant: <Response 1>" + 
    "\nUSER: " + d['conversation_a'][1]['content'] + 
    "\nAssistant: <Response 2>" for d in data]
cand1_texts = [
    "<Response 1>: " + d['conversation_a'][1]['content'] + 
    "\n<Response 2>: " + d['conversation_a'][3]['content'] for d in data]
cand2_texts = [
    "<Response 1>: " + d['conversation_b'][1]['content'] + 
    "\n<Response 2>: " + d['conversation_b'][3]['content'] for d in data]
for i, x in enumerate(data):
    new_item = {
        "id": f"mt_bench_human_judgments_{x['question_id']}",
        "instruction": inst,
        "input": inputs[i],
        "output": "",
        "candidates": [
            {
                "text": cand1_texts[i],
                "model": x['model_a'],
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": 1.0 if x['winner'] == 'model_a' else 0.0,
                }
            },
            {
                "text": cand2_texts[i],
                "model": x['model_b'],
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": 1.0 if x['winner'] == 'model_b' else 0.0,
                }
            }
        ]
    }
    new_data.append(new_item)
print(len(new_data))
with open("mt_bench_human_judgments.json", "w") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)