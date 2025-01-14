from baukit import TraceDict, get_module
import torch
import re
import bitsandbytes as bnb


def add_icv(edit_layer, icv, device, idx=-1):
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer == edit_layer:
            if isinstance(output, tuple):
                org_norm = torch.norm(output[0][:,idx])
                output[0][:, idx] += icv.to(device)
                output[0][:, idx] = output[0][:, idx] * (org_norm/torch.norm(output[0][:, idx]))
                return output
            else:
                return output
        else:
            return output

    return add_act


def icv_intervention(sentence, edit_layer, icv, model, model_config, tokenizer):

    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    MAX_NEW_TOKENS = 50

    output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, 
                    pad_token_id=tokenizer.eos_token_id, stop_strings = "\n\n", tokenizer=tokenizer).detach().cpu()
    clean_output = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])

    intervention_fn = add_icv(edit_layer, icv.reshape(1, model_config['resid_dim']), model.device, idx=-1)
    with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
        output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, 
                            pad_token_id=tokenizer.eos_token_id, stop_strings = "\n\n", tokenizer=tokenizer).detach().cpu()
        intervention_output = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])
    
    return clean_output, intervention_output

def add_icv_all(icv, device, idx=-1):
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if isinstance(output, tuple):
            org_norm = torch.norm(output[0][:,idx])
            output[0][:, idx] += icv[current_layer].to(device)
            output[0][:, idx] = output[0][:, idx] * (org_norm/torch.norm(output[0][:, idx]))
            return output
        else:
            return output
    return add_act

def icv_intervention_all_layer(sentence, icv, model, model_config, tokenizer):
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    MAX_NEW_TOKENS = 100

    output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, 
                    pad_token_id=tokenizer.eos_token_id, stop_strings = "\n\n", tokenizer=tokenizer).detach().cpu()
    clean_output = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])

    intervention_fn = add_icv_all(icv, model.device, idx=-1)
    with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
        output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, 
                            pad_token_id=tokenizer.eos_token_id, stop_strings = "\n\n", tokenizer=tokenizer).detach().cpu()
        intervention_output = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])
    
    return clean_output, intervention_output    