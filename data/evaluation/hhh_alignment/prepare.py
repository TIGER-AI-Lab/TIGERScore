from datasets import load_dataset
import json

dataset = load_dataset("HuggingFaceH4/hhh_alignment")
subsets = ['harmless', 'helpful', 'honest', 'other']
all_data = []
for subset in subsets:
    dataset = load_dataset("HuggingFaceH4/hhh_alignment", subset)
    data = dataset['test']
    instructions = [""] * len(data['input'])
    inputs = data['input']
    cand1_texts = [x['choices'][0] for x in data['targets']]
    cand2_texts = [x['choices'][1] for x in data['targets']]
    cand1_labels = [x['labels'][0] for x in data['targets']]
    cand2_labels = [x['labels'][1] for x in data['targets']]
    items = [
        {
            "id": f"hhh_alignment_{subset}_{i}",
            "instruction": instructions[i],
            "input": inputs[i],
            "output": "",
            "subset": subset,
            "candidates": [
                {
                    "text": cand1_texts[i],
                    "model": "unknown",
                    "decoding_method": "unknown",
                    "scores": {
                        "human_preference": cand1_labels[i],
                    }
                },
                {
                    "text": cand2_texts[i],
                    "model": "unknown",
                    "decoding_method": "unknown",
                    "scores": {
                        "human_preference": cand2_labels[i],
                    }
                }
            ]
        } for i in range(len(data['input']))
    ]
    all_data.extend(items)
    with open(f"{subset}.json", "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
        
        
with open("hhh_alignment.json", "w") as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)
print(len(all_data))