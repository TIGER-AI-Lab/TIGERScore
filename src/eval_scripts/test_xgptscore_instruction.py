# test set generate by Ref>ChatGPT>T5-XXL > T5-XL >T5-Large > T5-Base > T5-Small
import json,os,sys
# import pytrec_eval
import logging
from pathlib import Path
from itertools import chain
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.xgptscore import xgptscore
from utils import MyCorrelation
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json
logging.basicConfig(level=logging.INFO)
# from common.evaluation import overall_eval

dataset_root_path = "/home//WorkSpace/ExplainableGPTScore_bak/data"
dataset_name = "databricks/databricks-dolly-15k"
dataset_split = "test"
task='instruction-following'
model_list = ["ChatGPT","vicuna-33b-v1.3","vicuna-13b-v1.3","vicuna-7b-v1.3"] # ordered by model size
# not good for long-form QA, T5 series give short answer

dataset_root_path = Path(dataset_root_path)
dataset_file_path = dataset_root_path/dataset_name/f"{dataset_split}_data.json"
dataset_candidates_path = dataset_root_path/dataset_name/"candidates"/dataset_split/"top_p_sampling"
xgptscore_mode="instruction_following"
version_key=f"{xgptscore_mode}.old_2"
middle_file = dataset_root_path/dataset_name/f"rank_eval_mid.json"
output_file = dataset_root_path/dataset_name/f"xgptscore_eval.{version_key}.json"
overwrite = False
max_size = 200
model_name="ChatGPT"

if not middle_file.exists():
    # load dataset
    datas = json.load(open(dataset_file_path))
    # for item in datas:
    #     item["candidates"] = [
    #         {
    #             "text":item["output"] if isinstance(item["output"],str) else item["output"][0],
    #             "source":"ref"
    #         }
    #     ]
    for item in datas:
        item["candidates"] = []
    data_dict = {item["id"]:item for item in datas}
    # load candidates

    #TODO: we choose the cand with highest BARTScore
    for i,_model_name in enumerate(model_list):
        model_data = json.load(open(dataset_candidates_path /f"{_model_name}.eval.json"))
        for item in model_data:
            cands = sorted(item["candidates"], key=lambda x:x["scores"]["bart_score"], reverse=True)
            cands[0]["scores"]["rank"] = -i
            data_dict[item["id"]]["candidates"].append({
                "text":cands[0]["text"],
                "source":_model_name,
                "scores":cands[0]["scores"]
            })

    items = [data_dict[item_id] for item_id in data_dict.keys()]
    candidates = [[candidate["text"] for candidate in data["candidates"] ]for data in items]
    
    # targets = [data["output"] for data in items]
    # metrics = ["bleu", "rouge1","rouge2","rougeL","bart_score","bart_score_cnn"]
    # scores = overall_eval(candidates, targets, metrics, 1)
    # for metric in metrics:
    #     scores_metric = scores[metric]
    #     for i, sample in enumerate(items):
    #         _sample_cands = sample['candidates']
    #         for j, _sample_cand in enumerate(_sample_cands):
    #             _sample_cand['scores'][metric] = scores_metric[i][j]
    with open(middle_file, "w") as f:
        json.dump(items, f, indent=4)
        logging.info("Saved to {}".format(middle_file))
else:
    with open(middle_file, "r") as f:
        items = json.load(f)
    
# evaluate XGPTScore
if not output_file.exists() or overwrite:
    # eval_by_other_metrics
    # Data processing
    xgptitems = []
    for item in items:
        for cand in item['candidates']:
            xgptitems.append(XPGTItem(
                task=task,
                instruction=item['instruction'],
                input=item['input'],
                ref_output=item['output'],
                hypo_output=cand['text']
            ))
    # Run xgptscore
    result = xgptscore(xgptitems, mode=xgptscore_mode, model_name=model_name)
    idx = 0
    for item in items:
        for cand in item['candidates']:
            cand['responses'] = result['round_completions'][idx]
            cand['messages_records'] = result['messages_records'][idx]
            cand['scores']['xgptscore'] = get_xgptscore_from_json(cand['responses'][-1])
            idx += 1
        
    # Save results
    with open(output_file, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
        logging.info("Saved to {}".format(output_file))
else:
    logging.info("Loading existing results from {}".format(output_file))
    with open(output_file, "r") as f:
        items = json.load(f)
        
# evaluate system



num_cands = len(items[0]['candidates'])
human_scores = [[cand['scores']["rank"] for cand in item['candidates']] for item in items]
human_scores = list(chain(*zip(*human_scores))) # transpose and flatten
metrics = ["xgptscore","bleu","rouge1","rouge2","rougeL","rougeLsum","bart_score","bart_score_cnn"]
# metrics = ["xgptscore"]

Pearson_corr = {}
Spearman_corr = {}
Kendall_corr = {}
for metric in metrics:
    metric_scores = [[cand['scores'][metric] for cand in item['candidates']] for item in items]
    metric_scores = list(chain(*zip(*metric_scores))) # transpose and flatten
    metric_corr = MyCorrelation(num_cands, human_scores, metric_scores)
    Pearson_corr[metric] = metric_corr.Pearson()
    Spearman_corr[metric] = metric_corr.Spearman()
    Kendall_corr[metric] = metric_corr.Kendall()

# sort Corr
Pearson_corr = {k: v for k, v in sorted(Pearson_corr.items(), key=lambda item: item[1][0], reverse=True)}
Spearman_corr = {k: v for k, v in sorted(Spearman_corr.items(), key=lambda item: item[1][0], reverse=True)}
Kendall_corr = {k: v for k, v in sorted(Kendall_corr.items(), key=lambda item: item[1][0], reverse=True)}
Corr_record = {
    "Pearson": Pearson_corr,
    "Spearman": Spearman_corr,
    "Kendall": Kendall_corr,
}
# Save correlation results
corr_results_file = Path("./eval_results/") / (output_file.stem + ".corr.json")
corr_results_file.parent.mkdir(parents=True, exist_ok=True)
with open(corr_results_file, "w") as f:
    json.dump(Corr_record, f, indent=4, ensure_ascii=False)
    logging.info("Saved to {}".format(corr_results_file))
# save to another location
corr_results_file = output_file.parent / "eval_results" / (output_file.stem + ".corr.json")
corr_results_file.parent.mkdir(parents=True, exist_ok=True)
with open(corr_results_file, "w") as f:
    json.dump(Corr_record, f, indent=4, ensure_ascii=False)
    logging.info("Saved to {}".format(corr_results_file))
# print("Correlation results:")
# print(json.dumps(Corr_record, indent=4, ensure_ascii=False))