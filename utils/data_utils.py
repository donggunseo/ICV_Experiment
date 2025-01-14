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
    
