import torch.nn as nn
from tqdm import tqdm
import random
from utils.prompt_utils import *
import torch
from utils.intervention_utils import *
from sklearn.metrics import f1_score


def instruction_kv_caching(model, tokenizer, prefixes=None, separators=None):
    prompt = ""
    if prefixes['instructions']!="":
        prompt += prefixes['instructions'] + separators['instructions']
    device = model.device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values.detach()
    del outputs
    return past_key_values

def icl_without_intervention(train_dataset, test_dataset, n_shots, model, tokenizer,  
                             prefixes=None, separators=None, kv_cache=None):
    res = []
    gt = []
    pred = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        if n_shots == 0:
            demonstrations = []
        else:
            demonstrations = random.sample(train_dataset, n_shots)
        test_query = test_dataset[j]['input']
        prompt = create_prompt(demonstrations, test_query, prefixes, separators, insert_inst=True)
        device = model.device
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        test_target = test_dataset[j]['output']

        MAX_NEW_TOKENS = 50
        output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, stop_strings = "\n\n", tokenizer=tokenizer).detach().cpu()
        output_str = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])
        output_str = output_str.lower().strip()
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "gt" : test_target,
            "prediction" : output_str 
        })
        gt.append(test_target)
        if n_shots==0:
            if test_target in output_str:
                output_str = test_target
        pred.append(output_str)
    fs_f1 = f1_score(gt, pred, average='macro')
    return res, fs_f1


def icl_with_intervention(test_dataset, icv, model, model_config, tokenizer, prefixes, separators, eval_edit_layer=-1):
    res = []
    gt = []
    intervention_pred = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        test_query = test_dataset[j]['input']
        prompt = create_prompt([], test_query, prefixes, separators, insert_inst=True)
        test_target = test_dataset[j]['output']

        intervention_output = icv_intervention(prompt, eval_edit_layer, icv, model, model_config, tokenizer)
        
        intervention_output = intervention_output.lower().strip()
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "gt" : test_target,
            "intervention_prediction" : intervention_output
        })
        gt.append(test_target)
        intervention_pred.append(intervention_output)
    intervention_f1 = f1_score(gt, intervention_pred, average="macro")
    return res, intervention_f1


