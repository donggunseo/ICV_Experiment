from utils.data_utils import *
import random

def create_prompt(demonstrations, test_query, prefixes, separators):
    prompt = ""
    if prefixes['instructions']!="":
        prompt += prefixes['instructions'] + separators['instructions']

    for example in demonstrations:
        prompt+=prefixes['input']+" "+example['input']+separators['input']
        prompt+=prefixes['output']+" "+example['output']+separators['output']
    prompt+=prefixes['input']+" "+test_query+separators['input']+prefixes['output']
    return prompt

