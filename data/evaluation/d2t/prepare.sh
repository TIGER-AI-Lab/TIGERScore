# to webnlg_2020/test_data_prepared.json
cd webnlg_2020 && bash prepare.sh && cd .. 
# to webnlg_2017/test_data.json
python get_webnlg_eval_data.py --data_version 2017 && \
mv ./webnlg_2017/train_data.json ./webnlg_2017/test_data_prepared.json 