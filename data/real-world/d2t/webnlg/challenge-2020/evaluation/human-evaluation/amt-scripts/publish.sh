#!/bin/sh

while true
do    
    echo "publishing hits"
    
    python publish.py
    --hit_properties_file ./hit_utils/task_hit_properties.json
    --html_template ./hit_utils/task_layout.json
    --input_json_file ./data/en/inputs/hit_inputs.json
    --hit_ids_file ./data/en/inputs/hit_ids.txt
    --prod

done