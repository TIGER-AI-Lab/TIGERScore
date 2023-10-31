'''data-text pair generator'''

import os
import json
from glob import glob
import pandas as pd
import argparse
import time
import utils

def get_dt_pair(args):
    
    '''
    generate n_hits data-text pairs for evaluation
    structure for each saved HIT:
    [data image link, text, task, submission id, sample id, triplet size]
    '''
    
    image_link = args.config.get('image_link')
    table_basepath = args.config.get('table_basepath')
    text_basepath = args.config.get('text_basepath')
    
    task = args.task
    submission_id = args.submission_id
    triplet_size = args.triple_size
    hit_input_file  = './data/en/inputs/hit_inputs_' + task + '_' + submission_id + '_ts_' + triplet_size + '.json'
    
    # prepare hit input file
    if os.path.isfile(hit_input_file):
        pass
    else:
        with open(hit_input_file, 'w') as f9:
            json.dump({}, f9)
            
    fname = submission_id.split('id_')[1] + '_' + triplet_size
    with open(text_basepath + f'texts_{fname}.json', 'r') as f2:
        texts = json.load(f2)
    
    # load data and texts
    LINKS = []
    for file in glob(table_basepath + '*.jpg'):
        fn = file.split('/')[-1]
        sample_id = fn.split('.jpg')[0]
        if sample_id in texts.keys():
            data_link = image_link + fn
            text = texts[sample_id]
            LINKS.append([data_link, text, task, submission_id, sample_id, triplet_size])

    strs = [json.dumps(innerlist) for innerlist in LINKS]
    strs = "%s" % "\n".join(strs)
    with open(hit_input_file, 'w') as f6:
        f6.write(strs)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config_example.json',
                      type=utils.json_file)
    parser.add_argument('--task', type=str, default='rdf2textEN',
                       help='specify the task for evaluation')
    parser.add_argument('--submission_id', type=str, default='id_qual',
                       help='specify id of the submission that will be evaluated')
    parser.add_argument('--triple_size', type=str, default='1',
                       help='inputs are generated for triples of this size')
    args = parser.parse_args()
    
    get_dt_pair(args)
