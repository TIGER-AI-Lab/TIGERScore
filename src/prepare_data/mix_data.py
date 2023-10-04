import os,sys,json
import numpy as np
import shutil
mix_data = {
    "summarization": "/home//WorkSpace/ExplainableGPTScore_bak/data/sum/train_data_new.align_score.filter_v2.json",
    "data2text": "/home//WorkSpace/ExplainableGPTScore_bak/data/d2t/train_data_new.d2t.distill_new.filter_v3.json",
    "translation" : "/home//WorkSpace/ExplainableGPTScore_bak/data/wmt/train_data.wmt.distill.filter_v3.json",
    "long-form QA": "/home//WorkSpace/ExplainableGPTScore_bak/data/lfqa/train_data_new.longform_qa.v3.json",
    "instruction-following": "/home//WorkSpace/ExplainableGPTScore_bak/data/ins/train_data_new.instruction_following_new.v3.json"
}


for task in mix_data:
    # copy data to mix_data.json
    shutil.copy(mix_data[task], f"/home//WorkSpace/ExplainableGPTScore_bak/data/real_world/{task}.json")
        
        
# print(len(data))