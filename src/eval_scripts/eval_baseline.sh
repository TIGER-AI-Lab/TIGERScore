#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH -c 3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --output=../../jobs/%x/%j.out

metrics=("bleu" "rouge" "bertscore" "bleurt" "comet_da" "bart_score_cnn" "bart_score_para" "bart_score_cnn_src_hypo" "bart_score_para_src_hypo" "unieval_sum" "cometkiwi_da")

# summarization
input_file="../../BARTScore/SUM/SummEval/final_p_with_xgptscore.json"
output_file="../../BARTScore/SUM/SummEval/final_p_with_xgptscore.eval.json"
human_score_names="coherence,consistency,fluency,relevance"
cp -u $input_file $output_file
for metric in "${metrics[@]}"; do
    echo "Evaluating $metric"
    python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metric" \
        --human_score_names "$human_score_names"
done
python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metrics" \
    --human_score_names "$human_score_names" --print_results True

# data2text
input_file="../../data_bak/webnlg/webnlg2020_gen_with_scores.json"
output_file="../../data_bak/webnlg/webnlg2020_gen_with_scores.eval.json"
human_score_names="Correctness,DataCoverage,Fluency,Relevance,TextStructure"
cp -u $input_file $output_file
for metric in "${metrics[@]}"; do
    echo "Evaluating $metric"
    python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metric" \
        --human_score_names "$human_score_names"
done
python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metrics" \
    --human_score_names "$human_score_names" --print_results True

# long_form_QA
input_file="../../data_bak/lfqa/test.gpt-4.rank.json"
output_file="../../data_bak/lfqa/test.gpt-4.rank.eval.json"
human_score_names="rank"
cp -u $input_file $output_file
for metric in "${metrics[@]}"; do
    echo "Evaluating $metric"
    python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metric" \
        --human_score_names "$human_score_names"
done
python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metrics" \
    --human_score_names "$human_score_names" --print_results True

# instruction-following
input_file="../../data_bak/llm-blender/mix-instruct/test_data_prepared_300.json"
output_file="../../data_bak/llm-blender/mix-instruct/test_data_prepared_300.eval.json"
human_score_names="gpt_rank_score"
cp -u $input_file $output_file
for metric in "${metrics[@]}"; do
    echo "Evaluating $metric"
    python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metric" \
        --human_score_names "$human_score_names"
done
python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metrics" \
    --human_score_names "$human_score_names" --print_results True

# mathqa
input_file="../../data_bak/mathqa/gsm8k_test_output_prepared.json"
output_file="../../data_bak/mathqa/gsm8k_test_output_prepared.eval.json"
human_score_names="accuracy"
cp -u $input_file $output_file
for metric in "${metrics[@]}"; do
    echo "Evaluating $metric"
    python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metric" \
        --human_score_names "$human_score_names"
done
python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metrics" \
    --human_score_names "$human_score_names" --print_results True

# story_gen
input_file="../../data_bak/storygen/test.json"
output_file="../../data_bak/storygen/test.eval.json"
human_score_names="human"
cp -u $input_file $output_file
for metric in "${metrics[@]}"; do
    echo "Evaluating $metric"
    python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metric" \
        --human_score_names "$human_score_names"
done
python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metrics" \
    --human_score_names "$human_score_names" --print_results True

# translation 
input_file="../../data/wmt22/zh-en/eval_data.json"
output_file="../../data/wmt22/zh-en/eval_data.eval.json"
human_score_names="mqm"
cp -u $input_file $output_file
for metric in "${metrics[@]}"; do
    echo "Evaluating $metric"
    python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metric" \
        --human_score_names "$human_score_names"
done
python eval_baseline.py --input_file $output_file --output_file $output_file --metrics "$metrics" \
    --human_score_names "$human_score_names" --print_results True
