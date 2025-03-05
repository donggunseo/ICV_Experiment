from baukit import TraceDict, get_module
import torch
import re
from tqdm import tqdm
from utils.prompt_utils import *

def add_icv(edit_layer, icv, device, idx=-1):
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layer:
            if isinstance(output, tuple):
                print(len(output))
                print(output[0].shape)
                print(output[1])
                output[0][:, idx] += icv[current_layer].to(device)
                return output
            else:
                return output
        else:
            return output
    return add_act

def replace_icv(edit_layer, icv, device, idx=-1):
    def replace_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layer:
            if isinstance(output, tuple):
                output[0][:, idx] = icv[current_layer].to(device)
                return output
            else:
                return output
        else:
            return output
    return replace_act


def icv_intervention(sentence, edit_layer, icv, model, model_config, tokenizer, add=True, generate_str=False):
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    MAX_NEW_TOKENS = 500
    last_idx = inputs.input_ids.shape[1]
    if add:
        intervention_fn = add_icv(edit_layer, icv, device, idx=last_idx-1)
    else:
        intervention_fn = replace_icv(edit_layer, icv, device, idx=last_idx-1)

    if generate_str:
        def create_hook(layer_index, target_pos):
            def intervention_hook(module, inputs, output):
                # output: (batch_size, sequence_length, hidden_size)
                vec = icv[layer_index].to(output[0].device)
                # prompt의 마지막 토큰 위치에 intervention 벡터 더하기
                output[0][:, target_pos, :] += vec
                return output
            return intervention_hook

        # 각 transformer 레이어에 대해 hook 등록
        for i, layer in enumerate(model.model.layers):
            layer.register_forward_hook(create_hook(i, last_idx-1))


        # generate 과정에서 caching이 intervention 적용에 방해될 수 있으므로 캐시 비활성화
        model.config.use_cache = False
        
        
        max_new_tokens = 50
        generated = inputs.input_ids
        for _ in range(max_new_tokens):
            outputs = model(generated)
            logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
            next_token_logits = logits[:, -1, :]
            # 여기서는 간단히 argmax를 사용 (필요에 따라 sampling 등으로 변경 가능)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        output_str = tokenizer.decode(generated.squeeze()[len(inputs.input_ids.squeeze()):], skip_special_tokens = True)
        output_str = output_str.strip()
        intervention_output = output_str
    else:
        for _ in range(MAX_NEW_TOKENS):
            with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
                p_output = model.forward(inputs.input_ids)
                new_token_ids = torch.argmax(p_output.logits[:,-1,:])
                if new_token_ids == tokenizer.eos_token_id or new_token_ids==tokenizer.encode('\n\n', add_special_tokens=False)[0] or new_token_ids==tokenizer.encode('\n', add_special_tokens=False)[0]:
                    break
                new_token_ids = new_token_ids.view(1,-1).to(device)
                inputs.input_ids = torch.cat((inputs.input_ids, new_token_ids), dim=-1)
    intervention_output = tokenizer.decode(inputs.input_ids.squeeze()[last_idx:])
    
    return intervention_output


def icv_intervention_lens(sentence, edit_layer, icv, model, model_config, tokenizer, add=True):
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    MAX_NEW_TOKENS = 500
    last_idx = inputs.input_ids.shape[1]
    if add:
        intervention_fn = add_icv(edit_layer, icv, device, idx=last_idx-1)
    else:
        intervention_fn = replace_icv(edit_layer, icv, device, idx=last_idx-1)
    generated_traj = []
    for i in range(MAX_NEW_TOKENS):
        with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
            p_output = model.forward(inputs.input_ids)
        logits = torch.softmax(p_output.logits[:,-1,:], dim=-1).squeeze()
        sorted_indices = torch.argsort(logits, descending=True)
        lens = {tokenizer.decode(int(idx)): float(logits[idx]) for idx in sorted_indices[:10]}
        generated_traj.append(lens)
        new_token_ids = torch.argmax(p_output.logits[:,-1,:])
        if new_token_ids == tokenizer.eos_token_id or new_token_ids==tokenizer.encode('\n\n', add_special_tokens=False)[0] or new_token_ids==tokenizer.encode('\n', add_special_tokens=False)[0]:
            break
        new_token_ids = new_token_ids.view(1,-1).to(device)
        inputs.input_ids = torch.cat((inputs.input_ids, new_token_ids), dim=-1)
    intervention_output = tokenizer.decode(inputs.input_ids.squeeze()[last_idx:])
    
    return intervention_output, generated_traj










