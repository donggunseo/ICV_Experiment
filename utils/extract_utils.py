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

def get_mean_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)

        dummy_query = dummy_dataset[n]['input']
        prompt = create_prompt(demonstrations, dummy_query, prefixes, separators)

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with TraceDict(model, layers = model_config['layer_hook_names'], retain_input=True, retain_output=False) as td:
            model(**inputs)
        hs_by_layer = torch.vstack([td[layer].input.detach()[:,-1,:] for layer in model_config['layer_hook_names']])
        activation_storage[n]=hs_by_layer
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states


def get_diff_mean_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)

        dummy_query = dummy_dataset[n]['input']
        prompt = create_prompt(demonstrations, dummy_query, prefixes, separators)

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with TraceDict(model, layers = model_config['layer_hook_names'], retain_input=True, retain_output=False) as td:
            model(**inputs)
        hs_by_layer = torch.vstack([td[layer].input.detach()[:,-1,:] for layer in model_config['layer_hook_names']])
        activation_storage[n]=hs_by_layer

    activation_storage_zs = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        dummy_query = dummy_dataset[n]['input']
        prompt = create_prompt([], dummy_query, prefixes, separators)

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with TraceDict(model, layers = model_config['layer_hook_names'], retain_input=True, retain_output=False) as td:
            model(**inputs)
        hs_by_layer = torch.vstack([td[layer].input.detach()[:,-1,:] for layer in model_config['layer_hook_names']])
        activation_storage_zs[n]=hs_by_layer
    mean_hidden_states_few = activation_storage.mean(dim=0)
    mean_hidden_states_zs = activation_storage_zs.mean(dim=0)
    mean_hidden_states = mean_hidden_states_few - 0.1*mean_hidden_states_zs
    return mean_hidden_states

def get_attn_diff_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)

        dummy_query = dummy_dataset[n]['input']
        prompt = create_prompt(demonstrations, dummy_query, prefixes, separators)
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        output_dict = model(**inputs, output_attentions = True, output_hidden_states = True, return_dict = True)

        fs_hs_by_layer = torch.vstack([output_dict.hidden_states[layer+1].detach()[:,-1,:] for layer in range(model_config['n_layers'])])

        attn_by_layer = torch.vstack([output_dict.attentions[layer].detach()[:,:,-1,:] for layer in range(model_config['n_layers'])])
        attn_by_layer = attn_by_layer.mean(dim=1) ## average by head

        zs_prompt = create_prompt([],dummy_query,prefixes,separators)
        zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)
        zs_output_dict = model(**inputs, output_hidden_states = True, return_dict = True)

        zs_hs_by_layer = torch.vstack([zs_output_dict.hidden_states[layer+1].detach()[:,-1,:] for layer in range(model_config['n_layers'])])

        zs_length = zs_inputs.input_ids.shape[1]-1
        zs_attn_ratio = attn_by_layer[:,-zs_length:].sum(dim=-1)

        icv = (fs_hs_by_layer - (zs_attn_ratio * zs_hs_by_layer))/(1-zs_attn_ratio)
        activation_storage[n] = icv
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states
         






