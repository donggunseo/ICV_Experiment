from baukit import TraceDict, get_module
import torch
import re


def add_icv(edit_layer, icv, device, idx=-1):
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer in edit_layer:
            if isinstance(output, tuple):
                output[0][:, idx] += icv[current_layer].to(device)
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
    last_idx = inputs.input_ids.shape[1]
    intervention_fn = add_icv(edit_layer, icv, device, idx=last_idx-1)
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

# def add_icv_all(icv, device, ratio=None, idx=-1):
#     def add_act(output, layer_name):
#         current_layer = int(layer_name.split(".")[2])
#         if current_layer<=15:
#             org_norm = torch.norm(output[0][:,idx])
#             output[0][:, idx] = ratio[current_layer]*output[0][:, idx] + (1-ratio[current_layer])*icv[current_layer].to(device)
#             output[0][:, idx] = output[0][:, idx] * (org_norm/torch.norm(output[0][:, idx]))
#             return output
#         else:
#             return output
#     return add_act

# def icv_intervention_all_layer(sentence, icv, model, model_config, tokenizer, ratio=None):
#     device = model.device
#     inputs = tokenizer(sentence, return_tensors='pt').to(device)
#     MAX_NEW_TOKENS = 50
#     last_idx = inputs.input_ids.shape[1]
#     output = model.generate(**inputs, max_new_tokens = MAX_NEW_TOKENS, 
#                     pad_token_id=tokenizer.eos_token_id, stop_strings = ["\n\n", "\n", tokenizer.eos_token], tokenizer=tokenizer).detach().cpu()
#     clean_output = tokenizer.decode(output.squeeze()[len(inputs.input_ids.squeeze()):])

#     intervention_fn = add_icv_all(icv, device, ratio, idx=last_idx-1)
#     for _ in range(MAX_NEW_TOKENS):
#         with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
#             p_output = model.forward(inputs.input_ids)
#             new_token_ids = torch.argmax(p_output.logits[:,-1,:])
#             # new_token_prob = torch.softmax(p_output.logits[:,-1,:], dim=-1)
#             # _, top_k_indices = torch.topk(new_token_prob, 10)
#             # print("Top-10 Tokens and Probabilities:")
#             # for token in top_k_indices[0]:
#             #     token_str = tokenizer.decode([token.item()])  # 토큰 ID를 문자열로 변환
#             #     print(f"Token: {token_str}, Probability: {new_token_prob[:,token].item():.4f}")
#             if new_token_ids == tokenizer.eos_token_id or new_token_ids==271 or new_token_ids==198:
#                 break
#             new_token_ids = new_token_ids.view(1,-1).to(device)
#             inputs.input_ids = torch.cat((inputs.input_ids, new_token_ids), dim=-1)
#     intervention_output = tokenizer.decode(inputs.input_ids.squeeze()[last_idx:])
#     return clean_output, intervention_output    

