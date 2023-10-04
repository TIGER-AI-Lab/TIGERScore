data_dir="../../data/sum"
log_dir="logs/sum"
mkdir -p $log_dir
mkdir -p $data_dir
# SummEval, get the models' outputs
summeval_dir="${data_dir}/summeval"
mkdir -p $summeval_dir
for i in `seq 0 23`; do
    wget "https://storage.googleapis.com/sfr-summarization-repo-research/M${i}.tar.gz" -O "${summeval_dir}/M${i}.tar.gz"
    tar -xvf "${summeval_dir}/M${i}.tar.gz" -C "${summeval_dir}"
    rm "${summeval_dir}/M${i}.tar.gz"
done

# download cnn_stories and dailymail_stories from https://cs.nyu.edu/~kcho/DMQA/
gdown https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ -O "${summeval_dir}/cnn_stories.tgz"
gdown https://drive.google.com/uc?id=0BwmD_VLjROrfM1BxdkxVaTY2bWs -O "${summeval_dir}/dailymail_stories.tgz"
mkdir -p "${summeval_dir}/cnndm"

echo "Extracting cnn_stories.tgz and dailymail_stories.tgz"
echo "This may take a few minutes..., please wait."
tar -zxf "${summeval_dir}/cnn_stories.tgz" -C "${summeval_dir}/cnndm"
tar -zxf "${summeval_dir}/dailymail_stories.tgz" -C "${summeval_dir}/cnndm"
rm "${summeval_dir}/cnn_stories.tgz"
rm "${summeval_dir}/dailymail_stories.tgz"

wget "https://raw.githubusercontent.com/Yale-LILY/SummEval/master/data_processing/pair_data.py" -O "${summeval_dir}/pair_data.py"
for i in `seq 0 23`; do
    python3 ${summeval_dir}/pair_data.py --model_outputs "${summeval_dir}/M${i}" --story_files "${summeval_dir}"
done

# newsroom