# Download the BARTScore used system outputs and references
scripts_dir=$(pwd)
data_dir="../../data/bartscore_data"
mkdir -p $data_dir

# Summarization
cd $data_dir
datasets=("Newsroom" "QAGS_CNN" "QAGS_XSUM" "REALSumm" "Rank19" "SummEval")
mkdir -p summarization
for dataset in ${datasets[@]}; do
    wget "https://github.com/neulab/BARTScore/raw/main/SUM/${dataset}/data.pkl" -O "summarization/${dataset}.pkl"
done
cd $scripts_dir
python bartscore_data_process.py --data_dir "$data_dir" --task "summarization"


# Translation
cd $data_dir
datasets=("de-en" "fi-en" "gu-en" "kk-en" "lt-en" "ru-en" "zh-en")
mkdir -p translation
for dataset in ${datasets[@]}; do
    wget "https://github.com/neulab/BARTScore/raw/main/WMT/${dataset}/data.pkl" -O "translation/${dataset}.pkl"
done
cd $scripts_dir
python bartscore_data_process.py --data_dir "$data_dir" --task "translation"

# Data2Text
cd $data_dir
datasets=("BAGEL" "SFHOT" "SFRES")
mkdir -p data2text
for dataset in ${datasets[@]}; do
    wget "https://github.com/neulab/BARTScore/raw/main/D2T/${dataset}/data.pkl" -O "data2text/${dataset}.pkl"
done
cd $scripts_dir
python bartscore_data_process.py --data_dir "$data_dir" --task "data2text"