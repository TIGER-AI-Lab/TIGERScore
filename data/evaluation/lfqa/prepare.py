import json
from pathlib import Path
import logging
import fire


def main(middle_file: str, dataset_root_path: str, dataset_name: str, dataset_split: str, model_list : str):
    middle_file = Path(middle_file)
    dataset_root_path = Path(dataset_root_path)
    dataset_file_path = dataset_root_path / \
        dataset_name / f"{dataset_split}_data.json"
    dataset_candidates_path = dataset_root_path / dataset_name / \
        "candidates" / dataset_split / "top_p_sampling"
    model_list = model_list.split(",")
    if not middle_file.exists():
        # load dataset
        datas = json.load(open(dataset_file_path))
        # for item in datas:
        #     item["candidates"] = [
        #         {
        #             "text":item["output"] if isinstance(item["output"],str) else item["output"][0],
        #             "source":"ref"
        #         }
        #     ]
        for item in datas:
            item["candidates"] = []
        data_dict = {item["id"]: item for item in datas}
        # load candidates

        for i, _model_name in enumerate(model_list):
            model_data = json.load(
                open(dataset_candidates_path / f"{_model_name}.eval.json"))
            for item in model_data:
                cands = sorted(
                    item["candidates"], key=lambda x: x["scores"]["bart_score"], reverse=True)
                cands[0]["scores"]["rank"] = -i
                data_dict[item["id"]]["candidates"].append({
                    "text": cands[0]["text"],
                    "source": _model_name,
                    "scores": cands[0]["scores"]
                })

        items = [data_dict[item_id] for item_id in data_dict.keys()]
        with open(middle_file, "w") as f:
            json.dump(items, f, indent=4)
            logging.info("Saved to {}".format(middle_file))
    else:
        with open(middle_file, "r") as f:
            items = json.load(f)


if __name__ == "__main__":
    fire.Fire(main)
