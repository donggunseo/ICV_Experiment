import os
import json
from typing import *


def load_dataset(dataset_name):
    with open(f"./dataset_files/{dataset_name}/train.json", "r") as f:
        train = json.load(f)
    with open(f"./dataset_files/{dataset_name}/val.json", "r") as f:
        val = json.load(f)
    with open(f"./dataset_files/{dataset_name}/test.json", "r") as f:
        test = json.load(f)
    return train, val, test

def is_generate_sampling(dataset_name):
    if dataset_name in \
        [
            "xlsum",
            "wmt19", 
            "gsm8k", 
            'math_algebra', 
            'math_counting_and_probability', 
            'math_geometry', 
            'math_intermediate_algebra', 
            'math_number_theory', 
            'math_prealgebra', 
            'math_precalculus'
        ]:
        generate_str=True
    elif dataset_name in \
    ['next_item',
    'capitalize_first_letter',
    'choose_first_of_5',
    'english-french',
    'english-german',
    'park-country',
    'landmark-country',
    'english-spanish',
    'synonym',
    'country-capital',
    'singular-plural',
    'antonym'
    ]:
        generate_str=True
    else:
        generate_str=False
    return generate_str
