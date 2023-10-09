data_dir="../../data/d2t"
log_dir="logs/d2t"
mkdir -p $log_dir
mkdir -p $data_dir
# WebNLG, get the models' outputs
webnlg_dir="${data_dir}/webnlg"
mkdir -p $summeval_dir
python get_d2t_train_data.py 
