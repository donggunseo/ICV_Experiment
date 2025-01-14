import torch.nn as nn
from tqdm import tqdm
import random
from utils.prompt_utils import *
import torch
from utils.intervention_utils import *
from sklearn.metrics import f1_score

def icl_without_intervention(train_dataset, test_dataset, n_shots, model, model_config, tokenizer,  
                             prefixes=None, separators=None):
    res = []
    gt = []
    pred = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        
        if n_shots == 0:
            demonstrations = []
        else:
            demonstrations = random.sample(train_dataset, n_shots)
        
        test_query = test_dataset[j]['input']

        prompt = create_prompt(demonstrations, test_query, prefixes, separators)
        test_target = test_dataset[j]['output']
        
        output_str = sentence_eval(prompt, model, tokenizer)
        output_str = output_str.lower().strip()
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "gt" : test_target,
            "prediction" : output_str 
        })
        gt.append(test_target)
        pred.append(output_str)
    f1 = f1_score(gt, pred, average='macro')
    return res, f1

def sentence_eval(sentence, model, tokenizer):
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)

    MAX_NEW_TOKENS = 50
    output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, 
                            pad_token_id=tokenizer.eos_token_id, stop_strings = "\n\n", tokenizer=tokenizer).detach().cpu()
    output_str = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])
    return output_str


def icl_with_intervention(test_dataset, icv, model, model_config, tokenizer, eval_edit_layer, prefixes, separators):
    res = []
    gt = []
    zs_pred = []
    intervention_pred = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        
        test_query = test_dataset[j]['input']

        prompt = create_prompt([], test_query, prefixes, separators)
        test_target = test_dataset[j]['output']

        # zs_output, intervention_output = icv_intervention(prompt, eval_edit_layer, icv, model, model_config, tokenizer)
        zs_output, intervention_output = icv_intervention_all_layer(prompt, icv, model, model_config, tokenizer)
        
        zs_output = zs_output.lower().strip()
        intervention_output = intervention_output.lower().strip()
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "gt" : test_target,
            "zero_shot_prediction" : zs_output,
            "intervention_prediction" : intervention_output
        })
        gt.append(test_target)
        zs_pred.append(zs_output)
        intervention_pred.append(intervention_output)
    zs_f1 = f1_score(gt, zs_pred, average="macro")
    intervention_f1 = f1_score(gt, intervention_pred, average="macro")
    return res, zs_f1, intervention_f1


