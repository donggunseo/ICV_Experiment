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

def get_attn_diff_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples, N_TRIALS, prefixes=None, separators=None):
    dummy_dataset = random.sample(train_dataset, N_TRIALS)
    train_dataset_filtered = [item for item in train_dataset if item not in dummy_dataset]
    device = model.device
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])
    ratio_storage = torch.zeros(N_TRIALS, model_config['n_layers'])
    for n in tqdm(range(N_TRIALS)):
        demonstrations = random.sample(train_dataset_filtered, n_icl_examples)

        dummy_query = dummy_dataset[n]['input']
        prompt = create_prompt(demonstrations, dummy_query, prefixes, separators)
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        output_dict = model(**inputs, output_attentions = True, output_hidden_states = True, return_dict = True)

        fs_hs_by_layer = torch.vstack([output_dict.hidden_states[layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])

        attn_by_layer = torch.vstack([output_dict.attentions[layer].detach().cpu()[:,:,-1,:] for layer in range(model_config['n_layers'])])
        attn_by_layer = attn_by_layer.mean(dim=1) ## average by head

        del output_dict

        zs_prompt = create_prompt([],dummy_query,prefixes,separators)
        zs_inputs = tokenizer(zs_prompt, return_tensors='pt').to(device)
        zs_output_dict = model(**inputs, output_hidden_states = True, return_dict = True)

        zs_hs_by_layer = torch.vstack([zs_output_dict.hidden_states[layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])

        del zs_output_dict

        zs_length = zs_inputs.input_ids.detach().shape[1]-1
        zs_attn_ratio = attn_by_layer[:,-zs_length:].sum(dim=-1)

        icv = (fs_hs_by_layer - (zs_attn_ratio.unsqueeze(-1) * zs_hs_by_layer))/((torch.ones(zs_attn_ratio.shape[0])-zs_attn_ratio).unsqueeze(-1))
        activation_storage[n] = icv
        ratio_storage[n] = zs_attn_ratio
    mean_hidden_states = activation_storage.mean(dim=0)
    mean_ratio = ratio_storage.mean(dim=0)
    return mean_hidden_states, mean_ratio
         
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

        edit_layer = []
        for i in range(model_config['n_layers']):
            intervention_fn = add_icv(edit_layer, activation_storage[n], device, idx=-1)
            with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
                zs_hs = model(**zs_inputs, output_hidden_states = True, return_dict = True).hidden_states
                zs_hs_by_layer = torch.vstack([zs_hs[layer+1].detach().cpu()[:,-1,:] for layer in range(model_config['n_layers'])])
                del zs_hs
            activation_storage[n][i]=fs_hs_by_layer[i]-zs_hs_by_layer[i]
            edit_layer.append(i)
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


def add_activation_by_layer(inputs, model, model_config, fs_hs_by_layer):
    activation_storage = torch.zeros(model_config['n_layers'], model_config['resid_dim'])
    embedding_layer = model.model.embed_tokens
    decode_layers = model.model.layers
    rotary_emb = model.model.rotary_emb
    norm = model.model.norm
    device = model.device

    inputs_embeds = embedding_layer(inputs.input_ids)
    past_key_values = DynamicCache()
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
    position_ids = cache_position.unsqueeze(0)

    hidden_states = inputs_embeds

    position_embeddings = rotary_emb(hidden_states, position_ids)

    l_idx = 0
    for l in decode_layers:
        layer_outputs = l(
            hidden_states,
            attention_mask = None,
            position_ids = position_ids,
            past_key_values = past_key_values,
            output_attentions=False,
            use_cache=False,
            cache_position = cache_position,
            position_embeddings = position_embeddings,
        )
        hidden_states = layer_outputs[0]
        zs_hs = hidden_states[:,-1,:].squeeze().detach().cpu()
        icv_by_layer = fs_hs_by_layer[l_idx]-zs_hs
        activation_storage[l_idx] = icv_by_layer
        hidden_states[:,-1,:]+= icv_by_layer.unsqueeze(0).to(device)
        l_idx+=1
        del layer_outputs

    return activation_storage

        


        





