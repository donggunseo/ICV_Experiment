import torch.nn as nn
from tqdm import tqdm
import random
from utils.prompt_utils import *
import torch
from utils.intervention_utils import *
from sklearn.metrics import f1_score, accuracy_score
import re
from rouge import Rouge



def icl_without_intervention(train_dataset, test_dataset, n_shots, model, tokenizer,  
                             prefixes=None, separators=None, generate_str = False):
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

        MAX_NEW_TOKENS = 500
        if generate_str:
            output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, stop_strings = ["\n","\n\n",'<|eot_id|>'], tokenizer=tokenizer).detach().cpu()
            output_str = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])
            output_str = output_str.strip()
        else:
            output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, stop_strings = ["\n","\n\n",'<|eot_id|>'], tokenizer=tokenizer).detach().cpu()
            output_str = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])
            output_str = output_str.lower().strip()
        # output_str = re.sub(r'^\d+\.\s*', '', output_str)
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "gt" : test_target,
            "prediction" : output_str 
        })
        gt.append(test_target)
        pred.append(re.sub(r'^\d+\.\s*', '', output_str))
    if generate_str:
        rouge = Rouge()
        score = rouge.get_scores(pred, gt, avg=True)
    else:
        score = accuracy_score(gt, pred)
    return res, score


def icl_with_intervention(test_dataset, icv, model, model_config, tokenizer, prefixes, separators, eval_edit_layer=-1, add=True, generate_str = False):
    res = []
    gt = []
    intervention_pred = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        test_query = test_dataset[j]['input']
        prompt = create_prompt([], test_query, prefixes, separators, insert_inst=True)
        test_target = test_dataset[j]['output']

        intervention_output = icv_intervention(prompt, eval_edit_layer, icv, model, model_config, tokenizer, add, generate_str)
        
        if generate_str:
            intervention_output = intervention_output.strip()
        else:
            intervention_output = intervention_output.lower().strip()
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "gt" : test_target,
            "prediction" : intervention_output
        })
        gt.append(test_target)
        intervention_pred.append(re.sub(r'^\d+\.\s*', '', intervention_output))
    if generate_str:
        rouge = Rouge()
        intervention_score =rouge.get_scores(intervention_pred, gt, avg=True)
    else:
        intervention_score = accuracy_score(gt, intervention_pred)
    return res, intervention_score

def icl_with_intervention_lens(test_dataset, icv, model, model_config, tokenizer, prefixes, separators, add=True, generate_str=False):
    res = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        test_query = test_dataset[j]['input']
        prompt = create_prompt([], test_query, prefixes, separators, insert_inst=True)
        test_target = test_dataset[j]['output']
        lens = {}
        edit_layer = []
        for l in range(model_config['n_layers']):
            intervention_output, generated_lens = icv_intervention_lens(prompt, edit_layer, icv, model, model_config, tokenizer, add)
            intervention_output = intervention_output.lower().strip()
            lens[str(edit_layer)] = {'prediction':intervention_output, 'lens': generated_lens}
            edit_layer.append(l)
        intervention_output, generated_lens = icv_intervention_lens(prompt, edit_layer, icv, model, model_config, tokenizer, add)
        intervention_output = intervention_output.lower().strip()
        lens[str(edit_layer)] = {'prediction':intervention_output, 'lens': generated_lens}
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "gt" : test_target,
            "lens" : lens
        })
    return res


