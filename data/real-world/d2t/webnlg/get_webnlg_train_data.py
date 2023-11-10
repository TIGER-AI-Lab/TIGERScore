# webnlg2020
import json
import os

submissions_path = os.path.join(os.path.dirname(os.path.realpath(
    "__file__")), 'challenge-2020', 'submissions', 'rdf2text', 'en')
system_outputs = []
system_files = []
system_names = []
for root, dirs, files in os.walk(submissions_path):
    for system_ in dirs:
        system_files.append(
            open(os.path.join(root, system_, 'primary.en'), 'r').readlines())
        system_names.append(system_)

human_eval_files = {}
human_eval_path = os.path.join(os.path.dirname(os.path.realpath(
    "__file__")), 'challenge-2020', 'evaluation', 'human-evaluation', 'results', 'en')
for root, dirs, files in os.walk(human_eval_path):
    for system_ in dirs:
        if system_ in system_names:
            human_eval_files[system_] = json.load(
                open(os.path.join(root, system_, 'primary.json'), 'r'))

# both readlines
for line_num in range(len(system_files[0])):
    system_output = []
    for i, (model_name, model_output) in enumerate(zip(system_names, system_files)):
        scores_dict = {}
        if f"{line_num + 1}" in human_eval_files[model_name]:
            scores_dict = human_eval_files[model_name][f"{line_num + 1}"]
            if scores_dict:
                new_scores_dict = {k: 0 for k in list(
                    scores_dict.values())[0].keys()}
                if 'feedback' in new_scores_dict:
                    del new_scores_dict['feedback']
                for scores_one_dict in scores_dict.values():
                    for k, v in scores_one_dict.items():
                        if k != 'feedback':
                            new_scores_dict[k] += v
                for k, v in new_scores_dict.items():
                    new_scores_dict[k] = v / len(scores_dict)
                scores_dict = new_scores_dict

        system_output.append({'model': model_name,
                              "decoding_method": "greedy",
                              'text': model_output[line_num].strip(),
                              "scores": scores_dict
                              })
    system_outputs.append(system_output)
print(len(system_outputs[0]))
print(len(system_outputs))

reference_path = os.path.join(os.path.dirname(os.path.realpath(
    "__file__")), 'challenge-2020', 'evaluation', 'references', 'references-en.json')
references = json.load(open(reference_path, 'r'))['entries']
output_data = []
for i, item in enumerate(references):
    item = item[f"{i + 1}"]
    if not system_outputs[i][0]["scores"]:
        output_data.append({'id': i + 1,
                            'instruction': "Generate a description for the following triples.",
                            'input': "\n".join([f"({triple['subject']},{triple['property']},{triple['object']})" for triple in item["modifiedtripleset"]]).strip(),
                            'output': list(set([j["lex"] for j in item["lexicalisations"]])),
                            'candidates': system_outputs[i]
                            })
json.dump(output_data, open('webnlg2020_gen.json', 'w'),
          indent=4, ensure_ascii=False)

# output_data_with_scores = []
# for i,item in enumerate(references):
#     item = item[f"{i + 1}"]
#     if system_outputs[i][0]["scores"]:
#         output_data_with_scores.append({'id': i + 1,
#                         'instruction': "Generate a description for the following triples.",
#                         'input': "\n".join([f"({triple['subject']},{triple['property']},{triple['object']})" for triple in item["modifiedtripleset"]]).strip(),
#                         'output': list(set([j["lex"] for j in item["lexicalisations"]])),
#                         'candidates': system_outputs[i]
#                         })
# json.dump(output_data_with_scores, open('webnlg2020_gen_with_scores.json', 'w'), indent=4,ensure_ascii=False)
# print()
