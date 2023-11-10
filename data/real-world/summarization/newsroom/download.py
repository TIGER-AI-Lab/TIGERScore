from datasets import load_dataset
import fire
import json
from pathlib import Path

def main(
    dataset_name="Revankumar/News_room",
    split="train",
    max_size=3000,
    data_dir="./",
    shuffle=False,
    seed=42,
):
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
        
    new_data = []
    for i, item in enumerate(dataset.take(max_size)):
        new_item = {
            "id": item['url'],
            "instruction": "Write a summary of the text below.",
            "input": item['text'],
            "output": item['summary'],
            "data_source": dataset_name,
            "task": "summarization",
        }
        new_data.append(new_item)
    print("Final dataset size: {}".format(len(new_data)))
    
    task_ds_split_file = Path(data_dir) / f"{split}_data.json"
   
    with open(task_ds_split_file, "w") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
        print("Saved to {}".format(task_ds_split_file))
        
if __name__ == "__main__":
    fire.Fire(main)