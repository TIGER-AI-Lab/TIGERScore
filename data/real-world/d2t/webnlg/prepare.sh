git clone https://github.com/WebNLG/challenge-2020.git
python get_webnlg_train_data.py
mv webnlg2020_gen.json train_data.json
# webnlg2020_gen.json = webnlg2020_gen.json - webnlg2020_gen_with_scores.json
