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
from .intervention_utils import *
from transformers.cache_utils import DynamicCache


def get_mean_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)

        dummy_query = dummy_dataset[n]['input']
        prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=True)
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        hs = model(**inputs, output_hidden_states = True, return_dict = True).hidden_states

        fs_hs_by_layer = torch.vstack([hs[layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])
        activation_storage[n]=fs_hs_by_layer
        del hs
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
        fs_prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=True)
        fs_inputs = tokenizer(fs_prompt, return_tensors='pt').to(device)
        zs_prompt = create_prompt([], dummy_query, prefixes, separators, insert_inst=True)
        zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)

        fs_hs = model(**fs_inputs, output_hidden_states = True, return_dict = True).hidden_states
        fs_hs_by_layer = torch.vstack([fs_hs[layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])
        
        del fs_hs

        zs_hs = model(**zs_inputs, output_hidden_states = True, return_dict = True).hidden_states
        zs_hs_by_layer = torch.vstack([zs_hs[layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])

        del zs_hs

        activation_storage[n] = fs_hs_by_layer-zs_hs_by_layer
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states

ACTIVATION_STOR = []

def diff_act(edit_layer, act, device, idx=-1):
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layer:
            if isinstance(output, tuple):
                ACTIVATION_STOR.append(act[current_layer]-output[0][:, idx].detach().cpu())
                output[0][:, idx] = act[current_layer].to(device)
                return output
            else:
                return output
        else:
            return output
    return add_act

def get_diff_stacked_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)
        dummy_query = dummy_dataset[n]['input']
        fs_prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=True)
        fs_inputs = tokenizer(fs_prompt, return_tensors='pt').to(device)
        zs_prompt = create_prompt([],dummy_query,prefixes,separators, insert_inst=True)
        zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)

        fs_hs = model(**fs_inputs, output_hidden_states = True, return_dict = True).hidden_states
        fs_hs_by_layer = torch.vstack([fs_hs[layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])
        del fs_hs

        last_idx = zs_inputs.input_ids.shape[1]
        intervention_fn = diff_act(list(range(model_config['n_layers'])), fs_hs_by_layer, device, idx=last_idx-1)
        with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
            model(**zs_inputs)
        activation_storage[n]= torch.vstack(ACTIVATION_STOR)
        ACTIVATION_STOR.clear()
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states

def get_diff_stacked_hidden_states_filtered(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS*3)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    cnt = 0
    for n in tqdm(range(len(dummy_dataset))):
        if cnt==N_TRIALS:
            break
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)
        dummy_query = dummy_dataset[n]['input']
        dummy_target = dummy_dataset[n]['output']
        fs_prompt = create_prompt(demonstrations, dummy_query, prefixes, separators, insert_inst=True)
        fs_inputs = tokenizer(fs_prompt, return_tensors='pt').to(device)
        fs_output = model.generate(**fs_inputs, max_new_tokens = 50, pad_token_id=tokenizer.eos_token_id, stop_strings = "\n\n", tokenizer=tokenizer, return_dict_in_generate = True, output_hidden_states = True)
        if tokenizer.decode(fs_output.sequences.detach().cpu().squeeze()[len(fs_inputs.input_ids.squeeze()):]).lower().strip() == dummy_target:
            fs_hs_by_layer = torch.vstack([fs_output.hidden_states[0][layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])
            del fs_output
            del fs_inputs
            zs_prompt = create_prompt([],dummy_query,prefixes,separators, insert_inst=True)
            zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)
            act = add_activation_by_layer(zs_inputs, model, model_config, fs_hs_by_layer)
            activation_storage[cnt] = act
            cnt+=1
            print(f"few-shot correct. Progress {cnt}/{N_TRIALS}")
        else:
            del fs_output
            del fs_inputs
            print(f"few-shot fail. Progress {cnt}/{N_TRIALS}")
            continue
    print("N_Trials : ", cnt)
    mean_hidden_states = activation_storage.mean(dim=0)
    return mean_hidden_states



        







