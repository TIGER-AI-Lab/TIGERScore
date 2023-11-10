# Load Eval
log_dir="./wmt22/logs/"
mkdir -p $log_dir
wmt_version="wmt22"
for lang_pair in "zh-en" "en-de" "en-ru"; do
    python get_wmt_eval_data.py --human_score_name="mqm" --lang_pair="${lang_pair}" \
        --wmt_version="wmt22" 2>&1 | tee -a "${log_dir}/get_wmt22_eval_data_${lang_pair}.log"
done
for lang_pair in "en-zh"; do
    python get_wmt_eval_data.py --human_score_name="wmt-appraise-z" --lang_pair="${lang_pair}" \
        --wmt_version="wmt22" 2>&1 | tee -a "${log_dir}/get_wmt22_eval_data_${lang_pair}.log"
done
for lang_pair in "de-en" "ru-en"; do
    python get_wmt_eval_data.py --human_score_name="wmt-z" --lang_pair="${lang_pair}" \
        --wmt_version="wmt22" 2>&1 | tee -a "${log_dir}/get_wmt22_eval_data_${lang_pair}.log"
done