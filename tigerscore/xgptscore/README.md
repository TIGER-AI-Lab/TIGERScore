## XGPTScore Overview
This folder contains all the templates that we used to query ChatGPT or GPT-4 to get the identified errors in the hypothesis output for different tasks that TIGERScore involved. We call these API query methods as XGPTScore for a e**X**planainable **Scoring** method by querying **GPT** Models.

The overall pipeline of XGPTScore is:

1. We define a query template that askes GPT Models to idnetify errors in the hypothesis output based on the task instruction, source text and reference text.
2. We mannual construct various evaluation aspects to focus on for different tasks, as shown in [./constants.py](./constants.py).
3. Then, by applying the templates and also specifiy the aspects to focus on in the template, GPT Models are required to return the identified errors in a predefined format (like json format).

Sometimes GPTModels will output apparently lower-quality output if we require them to output in a specific format. To mitigate the affections from the predefined format on the response quality, we conduct 2-round evaluation. Firstly, we focus on the evaluation only, allowing the GPT models to output free-form evaluation results on the hypothesis output. Then we ask the GPT-models to format their free-form response in the first round into a specific format and provide elaborated information, which is an easier task for GPTModels.

## Quick start

We have provided a single function `xgptscore()` as the inferface, which takes the `xgptitems` along with the template mode and the OpenAI models as input to start the query.

Example Usage:
```python
task = "translation"
with open("example.json", "r") as f:
    items = json.load(f)
xgptitems = []
for item in items:
    for cand in item['candidates']:
        xgptitems.append(XPGTItem(
            task=task,
            instruction=item['instruction'],
            input=item['input'],
            ref_output=item['output'],
            hypo_output=cand['text']
        ))
result = xgptscore(xgptitems, "ea", "ChatGPT")
idx = 0
for item in items:
    for cand in item['candidates']:
        cand['responses'] = result['round_completions'][idx]
        cand['messages_records'] = result['messages_records'][idx]
json.dump(items, open("example_result.json", "w"), indent=4, ensure_ascii=False)
```

Please check out the input file `example.json` and the result file `example_results.json` to better understand how it actually works.