import json
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
task = "summarization"
task_data_map = {
    "summarization": "/home//WorkSpace/ExplainableGPTScore_bak/data/sum/train_data.align_score.distill.json"
}
with open(task_data_map[task], 'r') as f:
    data = json.load(f)

#filter
from fuzzywuzzy import fuzz
from copy import deepcopy
new_data = []
tmp_data = deepcopy(data)
# merge coherence and fluency
ratios = {
    'summeval': [],
    'xsum': [],
    'newsroom': [],
    'samsum': [],
}
ratios_75 = {
    'summeval': 64,
    'xsum': 40,
    'newsroom': 45,
    'samsum': 48,
} # get from numpy
ratios_50 = {
    'summeval': 78,
    'xsum': 62,
    'newsroom': 64,
    'samsum': 70,
}
for item in tmp_data:
    new_cands = []
    for cand in item['candidates']:
        put_in = True
        cand_scores = defaultdict(int)
        if isinstance(cand['responses'][-1], dict):
            for error in cand['responses'][-1]['errors'].values():
                if not isinstance(error, dict):
                    put_in = False
                for key in error.keys():
                    if key not in ["error_aspect", "severity", "explanation", "error_location", "score_reduction"]:
                        put_in = False
                        # print(key)
                for value in error.values():
                    if value is None:
                        put_in = False

                if error["error_aspect"] in ["Coherence", "Fluency", "Coherence, Fluency"]:
                    error["error_aspect"] = "Fluency"

                if error['score_reduction'] is None or error['score_reduction'] < 0:
                    put_in = False
                try:
                    float(error['score_reduction'])
                except:
                    put_in = False
                
                if error["error_aspect"] not in EVAL_ASPECTS["summarization"]:
                    put_in = False
                    # print(error["error_aspect"])
                if error["severity"] not in ["Major", "Minor"]:
                    put_in = False
                    # print(error["severity"])
                if error["explanation"] in ["", " ","N/A","None"]:
                    put_in = False
                    # print(error["explanation"])
                
                # if error["error_location"] is not None and error["error_location"] not in item["input"]:
                #     put_in = False
                #     # print(error["error_location"])

                
                # ratios[item["dataset"]].append(fuzz.token_set_ratio(item["input"], error["error_location"]))
                if fuzz.token_set_ratio(item["input"], error["error_location"]) < ratios_50[item["dataset"]]:
                    put_in = False
        else:
            put_in = False
        if put_in:
            new_cands.append(cand)
    if len(new_cands) > 0:
        item['candidates'] = new_cands
        new_data.append(item)

import numpy as np

for key,value in ratios.items():
    print(key,np.max(value),np.percentile(value,90),np.percentile(value,75),np.percentile(value,50),np.percentile(value,25),np.min(value))
    
print(f"Old data: {len(data)}")
print(f"Old data Candidates: {sum([len(item['candidates']) for item in data])}")
print(f"New data: {len(new_data)}")
print(f"New data Candidates: {sum([len(item['candidates']) for item in new_data])}")

new_output_file = "/home//WorkSpace/ExplainableGPTScore_bak/data/sum/train_data.align_score.filter.json"
with open(new_output_file, 'w') as f:
    json.dump(new_data, f, indent=4)