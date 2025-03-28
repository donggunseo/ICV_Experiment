import os, re, json
import random

import torch, numpy as np
import pandas as pd
from baukit import TraceDict
from tqdm import tqdm

# Include prompt creation helper functions
from .prompt_utils import *
from .inference_utils import *
from .model_utils import *

import torch.nn.functional as F


def get_mean_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None, insert_inst = True):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)

        dummy_query = dummy_dataset[n]['input']
        prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=insert_inst)
        
        fs_inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with TraceDict(model, layers = model_config['layer_hook_names'], retain_output=True) as td:
            model(**fs_inputs)
        fs_hs_by_layer = torch.vstack([td[l].output[0][:,-1,:].detach().cpu() for l in model_config['layer_hook_names']])
        del td
        del fs_inputs
        activation_storage[n]=fs_hs_by_layer

    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states


def get_diff_mean_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None, insert_inst=True):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)
        dummy_query = dummy_dataset[n]['input']
        fs_prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=insert_inst)
        fs_inputs = tokenizer(fs_prompt, return_tensors='pt').to(device)
        zs_prompt = create_prompt([], dummy_query, prefixes, separators, insert_inst=insert_inst)
        zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)

        with TraceDict(model, layers = model_config['layer_hook_names'], retain_output=True) as td:
            model(**fs_inputs)
        fs_hs_by_layer = torch.vstack([td[l].output[0][:,-1,:].detach().cpu() for l in model_config['layer_hook_names']])
        del td
        del fs_inputs

        with TraceDict(model, layers = model_config['layer_hook_names'], retain_output=True) as td:
            model(**zs_inputs)
        zs_hs_by_layer = torch.vstack([td[l].output[0][:,-1,:].detach().cpu() for l in model_config['layer_hook_names']])
        del td
        del zs_inputs

        activation_storage[n] = fs_hs_by_layer-zs_hs_by_layer
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states

ACTIVATION_STOR = []

def diff_act(edit_layer, act, device, idx=-1):
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layer:
            diff = act[current_layer]-output[0][:, idx].detach().cpu()
            ACTIVATION_STOR.append(diff)
            output[0][:, idx] += diff.to(device)
            return output
        else:
            return output
    return add_act

def get_diff_stacked_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None, insert_inst=True):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)
        dummy_query = dummy_dataset[n]['input']
        fs_prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=insert_inst)
        fs_inputs = tokenizer(fs_prompt, return_tensors='pt').to(device)
        zs_prompt = create_prompt([],dummy_query,prefixes, separators, insert_inst=insert_inst)
        zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            with TraceDict(model, layers = model_config['layer_hook_names'], retain_output=True) as td:
                model(**fs_inputs)
        fs_hs_by_layer = torch.vstack([td[l].output[0][:,-1,:].detach().cpu() for l in model_config['layer_hook_names']])
        del td
        del fs_inputs
        last_idx = zs_inputs.input_ids.shape[1]
        intervention_fn = diff_act(list(range(model_config['n_layers'])), fs_hs_by_layer, device, idx=last_idx-1)
        with torch.no_grad():
            with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
                model(**zs_inputs)
        activation_storage[n]= torch.vstack(ACTIVATION_STOR)
        ACTIVATION_STOR.clear()
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states

def get_diff_stacked_hidden_states_iterate(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    lens_agg = []
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)
        dummy_query = dummy_dataset[n]['input']
        fs_prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=True)
        fs_inputs = tokenizer(fs_prompt, return_tensors='pt').to(device)
        zs_prompt = create_prompt([],dummy_query,prefixes, separators, insert_inst=True)
        zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)

        with TraceDict(model, layers = model_config['layer_hook_names'], retain_output=True) as td:
            model(**fs_inputs)
        fs_hs_by_layer = torch.vstack([td[l].output[0][:,-1,:].detach().cpu() for l in model_config['layer_hook_names']])
        del td
        del fs_inputs
        fs_logits = torch.softmax(model.lm_head(fs_hs_by_layer[-1].to(device)),dim=-1)
        sorted_indices = torch.argsort(fs_logits, descending=True)
        fs_lens = {tokenizer.decode(int(idx)): float(fs_logits[idx]) for idx in sorted_indices[:10]}
        last_idx = zs_inputs.input_ids.shape[1]
        edit_layer=[]
        icv_storage = torch.zeros(model_config['n_layers'], model_config['resid_dim'])
        interv_lens = {}
        for l in range(model_config['n_layers']):
            intervention_fn = add_icv(edit_layer, icv_storage, device, idx=last_idx-1)
            with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn, retain_output=True) as td:
                model.forward(**zs_inputs)
            diff = fs_hs_by_layer[l]-td[model_config['layer_hook_names'][l]].output[0][:,-1,:].detach().cpu()
            icv_storage[l]=diff
            logits = torch.softmax(model.lm_head(td[model_config['layer_hook_names'][-1]].output[0][:,-1,:]), dim=-1).squeeze()
            sorted_indices = torch.argsort(logits, descending=True)
            lens = {tokenizer.decode(int(idx)): float(logits[idx]) for idx in sorted_indices[:10]}
            interv_lens[str(edit_layer)]={'kl-divergence':F.kl_div(logits.log(), fs_logits, reduction='sum').item(), 'output':lens}
            edit_layer.append(l)
            del td
        with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn, retain_output=True) as td:
            model.forward(**zs_inputs)
        logits = torch.softmax(model.lm_head(td[model_config['layer_hook_names'][-1]].output[0][:,-1,:]), dim=-1).squeeze()
        sorted_indices = torch.argsort(logits, descending=True)
        lens = {tokenizer.decode(int(idx)): float(logits[idx]) for idx in sorted_indices[:10]}
        interv_lens[str(edit_layer)]={'kl-divergence':F.kl_div(logits.log(), fs_logits, reduction='sum').item(), 'output':lens}
        del td
        lens_agg.append(
            {
                'fs_prompt':fs_prompt,
                'query' : dummy_query,
                'target' : dummy_dataset[n]['output'],
                'fs_lens' : fs_lens,
                'intervention_lens' : interv_lens
            }
        )
        activation_storage[n] = icv_storage
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states, lens_agg






        







