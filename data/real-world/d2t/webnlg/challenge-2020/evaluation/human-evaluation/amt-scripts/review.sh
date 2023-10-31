#!/bin/sh

count=0
    
while (( count < 100000 ));

    do

    echo 'results'

    python get_results.py
    --hit_ids_file ./01/inputs/hit_ids_4567_merged.txt
    --output_file ./01/out/hit_out_rdf2textEN_4567_merged.json
    --prod

    echo 'qual'

    #python approve_hits.py
    #--hit_ids_file ./01/inputs/hit_ids_4567_merged.txt
    --prod

    python hitcount_assign.py
    --hit_ids_file ./01/inputs/hit_ids_4567_merged.txt
    --submission_id 4567_merged
    --prod

    (( count++ ))
    done


