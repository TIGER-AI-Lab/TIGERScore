# %%
import json
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
task = "long-form QA"
task_data_map = {
    "long-form QA": "/home//WorkSpace/ExplainableGPTScore_bak/data/lfqa/train_data_new.longform_qa.v3.json"
}
with open(task_data_map[task], 'r') as f:
    data = json.load(f)
# all_file = "/home//WorkSpace/ExplainableGPTScore_bak/data/sum/train_data_all.json"
# with open(all_file, 'r') as f:
#     all_data = json.load(f)
# for item in data:
#     for _item in all_data:
#         if item["input"] == _item["input"]:
#             item["dataset"] = _item["dataset"]
#             break
# with open(task_data_map[task], 'w') as f:
#     json.dump(data, f, indent=4)

# %%
import sys
sys.path.append("/home//WorkSpace/ExplainableGPTScore_bak/src")
from xgptscore.constants import EVAL_ASPECTS
xgptscore_types = ['xgptscore',"Clarity",'Accuracy',"Fluency"]
# xgptscore_types = xgptscore_types + list(EVAL_ASPECTS[task].keys())
# xgptscore_types.remove('Coherence')
print(xgptscore_types)
scores = defaultdict(list)
datasets = defaultdict(int)
datasets_can = defaultdict(int)
penlty_distribution = defaultdict(int)
penlty_distribution_per_dataset = defaultdict(lambda: defaultdict(int))
score_distribution = defaultdict(int)
score_distribution_per_aspect = defaultdict(lambda: defaultdict(int))
for item in data:
    datasets[item['dataset']] += 1
    for cand in item['candidates']:
        datasets_can[item['dataset']] += 1
        cand_scores = defaultdict(int)
        penlty_num = 0
        if isinstance(cand['responses'][-1], dict):
            for error in cand['responses'][-1]['errors'].values():
                if error['score_reduction']: # is Not None and is not 0
                    cand_scores[error['error_aspect']] -= error['score_reduction']
                    cand_scores['xgptscore'] -= error['score_reduction']
                    penlty_num += 1
                    score_distribution[error['score_reduction']] += 1
                    score_distribution_per_aspect[error['error_aspect']][error['score_reduction']] += 1
        for key in xgptscore_types:
            scores[key].append(cand_scores[key])
        penlty_distribution[penlty_num] += 1
        penlty_distribution_per_dataset[item['dataset']][penlty_num] += 1
        

        
fig, axes = plt.subplots(len(xgptscore_types) // 2 + 1, 2)
for i, key in enumerate(scores):
    ax = axes[i // 2, i % 2]
    # make bins' width wider 
    ax.hist(scores[key], bins=np.arange(min(scores[key]), max(scores[key]) + 1, 1))
    print(min(scores[key]), max(scores[key]))
    ax.set_title(key)

plt.tight_layout()
plt.show()
print(datasets)
print(datasets_can)
# print(penlty_distribution_per_dataset)
print("score distribution:",Counter(score_distribution))
for key in score_distribution_per_aspect:
    print(key,":",Counter(score_distribution_per_aspect[key]))
print(Counter([x for x in scores['xgptscore']]))
print("penlty distribution",Counter(penlty_distribution))
# print("penlty distribution",penlty_distribution_per_dataset)
print("total candidates",len([x for x in scores['xgptscore']]))
print("0 ratio",len([x for x in scores['xgptscore'] if x == 0]) / len(scores['xgptscore']))

# %%
print(data[0])

# %%
#filter
from fuzzywuzzy import fuzz
from copy import deepcopy
new_data = []
tmp_data = deepcopy(data)
# merge coherence and fluency
ratios = defaultdict(list)
# ratios_75 = {
#     'totto': 47,
#     'kasnerz/wikitabletext': 42,
#     'webnlg': 21,
# }
# ratios_50 = {
#     'totto': 68.0,
#     'kasnerz/wikitabletext': 73,
#     'webnlg': 35,
# }
need_to_remove = 0
for item in tmp_data:
    new_cands = []
    for cand in item['candidates']:
        put_in = True
        cand_scores = defaultdict(int)
        if isinstance(cand['responses'][-1], dict):
            if len(cand['responses'][-1]['errors'].values()) > 10:
                put_in = False
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
                
                if error["error_aspect"] not in ["Clarity",'Accuracy',"Fluency"]:
                    put_in = False
                    # print(error["error_aspect"])
                if error["severity"] not in ["Major", "Minor"]:
                    put_in = False
                    # print(error["severity"])
                if error["explanation"] in ["", " ","N/A","None"]:
                    put_in = False
                    # print(error["explanation"])
                if error["score_reduction"] > 5:
                    put_in = False
                    # print(error["score_reduction"])
                if error["severity"] == "Major" and error["score_reduction"] < 2:
                    put_in = False
                # # if error["error_location"] is not None and error["error_location"] not in cand["text"]:
                #     put_in = False
                #     # print(error["error_location"])
                # if error["score_reduction"] <= 1:
                #     # ratios[item["dataset"]].append(fuzz.token_set_ratio(cand["text"], error["error_location"]))
                #     ratios[item["dataset"]].append(fuzz.o(cand["text"], error["explanation"]))
                    # if fuzz.token_set_ratio(cand["text"], error["error_location"]) < 100:
                    #     put_in = False
                #     put_in = False
                #     # print(error["error_location"])
                # if fuzz.token_set_ratio(cand["text"], error["error_location"]) < 50:
                #     put_in = False
                if error["severity"] == "Minor":
                    ratios[item["dataset"]].append(fuzz.token_set_ratio(cand["text"], error["error_location"]))
            # if len(cand['responses'][-1]['errors'].values()) in [1,2]:
            # #     error = list(cand['responses'][-1]['errors'].values())[0]
            # #     if error["error_aspect"] == "Accuracy" and error["severity"] == "Minor":、
            #     for error in cand['responses'][-1]['errors'].values():
            #         ratios[item["dataset"]].append(fuzz.token_set_ratio(cand["text"], error["error_location"]))
                # if not error["error_aspect"] == "Fluency":
                    # if fuzz.token_set_ratio(cand["text"], error["error_location"]) < 100:
                    #     put_in = False
        else:
            put_in = False
        if put_in:
            new_cands.append(cand)
    if len(new_cands) > 0:
        item['candidates'] = new_cands
        new_data.append(item)
    
print(f"Old data: {len(data)}")
print(f"Old data Candidates: {sum([len(item['candidates']) for item in data])}")
print(f"New data: {len(new_data)}")
print(f"New data Candidates: {sum([len(item['candidates']) for item in new_data])}")
print(need_to_remove)

# new_output_file = "/home//WorkSpace/ExplainableGPTScore_bak/data/lfqa/train_data_new.longform_qa.clean.json"
# with open(new_output_file, 'w') as f:
#     json.dump(new_data, f, indent=4)

# %%
from copy import deepcopy
import numpy as np
new_data = []
tmp_data = deepcopy(data)
np.random.seed(42)

# ratios_50 = {
#     'totto': 80.0,
#     'kasnerz/wikitabletext': 84,
#     'webnlg': 43,
# }


ratios_50 = {
    'din0s/asqa': 85,
    'Tingle/FeTaQA': 85,
    'cosmos_qa': 75,
    'eli5': 50,
}

for item in tmp_data:
    new_cands = []
    for cand in item['candidates']:
        need_to_remove = False
        for error in cand['responses'][-1]['errors'].values():
            if error["error_aspect"] != "Fluency":
                if fuzz.token_set_ratio(cand["text"], error["error_location"]) < ratios_50[item["dataset"]]:
                    need_to_remove = True
        if cand['scores']['xgptscore'] == 0 and np.random.random() < 1/2:
            continue
        # if len(cand['responses'][-1]['errors'].values()) == 1:
        #     error = list(cand['responses'][-1]['errors'].values())[0]
        #     if error["error_aspect"] != "Fluency" and error["score_reduction"] <= 1 and np.random.random() < 1/2:
        #         continue
        #     # if error["score_reduction"] <= 1:
        #     #     if fuzz.token_set_ratio(cand["text"], error["error_location"]) < 100:
        #     #         continue
        
        if not need_to_remove:
            new_cands.append(cand)
    if len(new_cands) == 0:
        continue
    item['candidates'] = new_cands
    new_data.append(item)
            
    
    
print(f"Old data: {len(data)}")
print(f"Old data Candidates: {sum([len(item['candidates']) for item in data])}")
print(f"New data: {len(new_data)}")
print(f"New data Candidates: {sum([len(item['candidates']) for item in new_data])}")

new_output_file = "/home//WorkSpace/ExplainableGPTScore_bak/data/lfqa/train_data_new.longform_qa.v3.json"
with open(new_output_file, 'w') as f:
    json.dump(new_data, f, indent=4)

# %%
import numpy as np


for key,value in ratios.items():
    print(key,len(value))
    print(key,np.max(value),np.percentile(value,90),np.percentile(value,75),np.percentile(value,50),np.percentile(value,25),np.percentile(value,10),np.min(value))


