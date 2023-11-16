import sys
sys.path.append("..")
from xgptscore.openai_utils import openai_completions, _chatml_to_prompt
import fire
import json
import random
from string import Template


template = """
Instruction:
${instruction}
${input}

A ground-truth response:
${output}

A model will be asked to respond to this instruction. However, that response might contain errors in various aspects such as helpfulness, harmfulness, honestness, hallucination, Logical conflicts, reasoning errors, misunderstanding context, bad output formats, etc. 

Please first output 5 possible error aspects if a model is asked to generate a response for the above instruction. The error aspects don't have to be one of the above aspects and can be any aspect that you think is reasonable for this instruction.

Then generate an incorrect response contains up to ${num_errors} errors of these aspects. Each error corresponds to one of the aspect.
The incorrect response should mimic style the real-generation of a model. 

Then give an analysis of these errors. For each error, give me the 
- error location (the substring that is wrong in the generated incorrect output)
- error aspect
- explanation (the generic error type description, why it's an error, and the correction suggestions)
- severity ("major" or "minor")
- score reduction (an integer between 1 to 5 given the severity of the error)

Output format:
Generated incorrect output: 

Error location 1:
Error aspect 1:
Explanation 1:
Severity 1:
Score reduction 1:
...
"""

def main(
    input_file, output_file, 
    model_name="gpt-4", num_samples=None, 
    num_procs=5, seed=42):
    random.seed(seed)
    with open(input_file, "r") as f:
        if input_file.endswith(".jsonl"):
            input_data = [json.loads(line) for line in f]
        elif input_file.endswith(".json"):
            input_data = json.load(f)
    if num_samples is None:
        num_samples = len(input_data)
    print(num_samples)
    input_data = input_data[:num_samples]
    
    def process_data(item):
        prompt = Template(template=template).substitute(
            instruction=item["instruction"],
            input=item["input"],
            output=item["output"],
            num_errors=random.randint(1, 5)
        )
        message = [{
            "role": "user",
            "content": prompt
        }]
        chatml_prompt = _chatml_to_prompt(message)
        return chatml_prompt
        
    prompts = list(map(process_data, input_data))
    print(prompts[0])
    completions = openai_completions(prompts, model_name=model_name, num_procs=num_procs, use_cache=True)
    print(f"Finished generating {len(completions['completions'])} completions.")
    print(f"Total prices: {sum(completions['price_per_example'])}")
    for i, completion in enumerate(completions['completions']):
        input_data[i]["completion"] = completion
    with open(output_file, "w") as f:
        if output_file.endswith(".jsonl"):
            for item in input_data:
                json.dump(item, f)
                f.write("\n")
        elif output_file.endswith(".json"):
            json.dump(input_data, f)
        
if __name__ == "__main__":
    fire.Fire(main)
    