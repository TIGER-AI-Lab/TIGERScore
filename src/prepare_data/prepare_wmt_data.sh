
# Translation
# Load Train
log_dir="logs/translation"
mkdir -p $log_dir
wmt_max_ex_size_per_lang_pair=2000
for lang_pair in "zh-en" "en-de" "en-ru"; do
    python get_wmt_train_data.py --human_score_name="mqm" --lang_pair="${lang_pair}" \
         --max_ex_size=$wmt_max_ex_size_per_lang_pair \
        2>&1 | tee -a "${log_dir}/get_wmt_train_data_${lang_pair}.log"
done
for lang_pair in "en-zh" "de-en" "ru-en"; do
    python get_wmt_train_data.py --human_score_name="da" --lang_pair="${lang_pair}" \
         --max_ex_size=$wmt_max_ex_size_per_lang_pair \
        2>&1 | tee -a "${log_dir}/get_wmt_train_data_${lang_pair}.log"
done
# Load Eval
wmt_version="wmt22"
for lang_pair in "zh-en" "en-de" "en-ru"; do
    python get_wmt_eval_data.py --human_score_name="mqm" --lang_pair="${lang_pair}" \
         --max_ex_size=$wmt_max_ex_size_per_lang_pair --wmt_version="wmt22"
        2>&1 | tee -a "${log_dir}/get_wmt22_eval_data_${lang_pair}.log"
done
for lang_pair in "en-zh"; do
    python get_wmt_eval_data.py --human_score_name="wmt-appraise-z" --lang_pair="${lang_pair}" \
         --max_ex_size=$wmt_max_ex_size_per_lang_pair --wmt_version="wmt22"
        2>&1 | tee -a "${log_dir}/get_wmt22_eval_data_${lang_pair}.log"
done
for lang_pair in "de-en" "ru-en"; do
    python get_wmt_eval_data.py --human_score_name="wmt-z" --lang_pair="${lang_pair}" \
         --max_ex_size=$wmt_max_ex_size_per_lang_pair --wmt_version="wmt22"
        2>&1 | tee -a "${log_dir}/get_wmt22_eval_data_${lang_pair}.log"
done

# aggregation
python aggregate_wmt_train_data.py --data_dir="../../data/wmt" \
    2>&1 | tee -a "${log_dir}/aggregate_wmt_train_data.log"


