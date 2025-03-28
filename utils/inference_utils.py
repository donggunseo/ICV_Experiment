import torch.nn as nn
from tqdm import tqdm
import random
from utils.prompt_utils import *
import torch
import re
from sklearn.metrics import f1_score, accuracy_score
from rouge import Rouge
import sacrebleu
from baukit import TraceDict
from utils.postprocess_utils import *


def n_shot_eval(inputs, model, tokenizer, generate_str=False, max_new_tokens=500, kv_cache = None, stop_strings = None):
    device = model.device
    inputs = inputs.to(device)
    og_length = len(inputs.input_ids.squeeze())-1 if kv_cache is not None else len(inputs.input_ids.squeeze())

    if generate_str:
        output = model.generate(**inputs, temperature = 0.8, top_p = 0.95, do_sample = True, max_new_tokens = max_new_tokens, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, past_key_values = kv_cache, stop_strings=stop_strings).detach().cpu()
        output_str = tokenizer.decode(output.squeeze()[og_length:], skip_speical_tokens=True)
        output_str = output_str.strip()
    else:
        output = model.generate(**inputs, max_new_tokens = max_new_tokens, pad_token_id=tokenizer.eos_token_id, eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer, past_key_values = kv_cache, stop_strings=stop_strings, do_sample=False).detach().cpu()
        output_str = tokenizer.decode(output.squeeze()[og_length:], skip_speical_tokens=True)
        output_str = output_str.strip()
    return output_str

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


def n_shot_eval_intervention(sentence, edit_layer, icv, model, model_config, tokenizer, add=True, generate_str=False, max_new_tokens = 500):
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    last_idx = inputs.input_ids.shape[1]-1
    
    if add:
        intervention_fn = add_icv(edit_layer, icv, device, idx=last_idx)
    else:
        intervention_fn = replace_icv(edit_layer, icv, device, idx=last_idx)

    with torch.no_grad():
        with TraceDict(model, layers = model_config['layer_hook_names'], edit_output=intervention_fn):
            output = model.forward(**inputs)
        new_token_ids = torch.argmax(output.logits[:,-1,:], dim=-1)
        kv_cache_intervention = output.past_key_values
        new_token_ids = new_token_ids.view(1,-1).to(device)
        inputs['input_ids'] = torch.cat((inputs.input_ids, new_token_ids), dim=-1)
        inputs['attention_mask'] = torch.cat((inputs.attention_mask, torch.tensor([[1]]).to(device)), dim=-1)
        del output
        
        output_str = n_shot_eval(inputs, model, tokenizer, generate_str=generate_str, max_new_tokens = max_new_tokens, kv_cache = kv_cache_intervention)
    return output_str

def post_process(prediction, dataset_name):
    if dataset_name == 'trec':
        id2label = {
            0: "abbreviation:abbreviation",
            1: "abbreviation:expression",
            2: "entity:animal",
            3: "entity:body",
            4: "entity:color",
            5: "entity:creative work",
            6: "entity:currency",
            7: "entity:disease and medicine",
            8: "entity:event",
            9: "entity:food",
            10: "entity:instrument",
            11: "entity:language",
            12: "entity:letter",
            13: "entity:other",
            14: "entity:plant",
            15: "entity:product",
            16: "entity:religion",
            17: "entity:sport",
            18: "entity:substance",
            19: "entity:symbol",
            20: "entity:technique and method",
            21: "entity:equivalent term",
            22: "entity:vehicle",
            23: "entity:special word",
            24: "description:definition",
            25: "description:description",
            26: "description:manner",
            27: "description:reason",
            28: "human:group",
            29: "human:individual",
            30: "human:title",
            31: "human:description",
            32: "location:city",
            33: "location:country",
            34: "location:mountain",
            35: "location:other",
            36: "location:state",
            37: "numeric:code",
            38: "numeric:count",
            39: "numeric:date",
            40: "numeric:distance",
            41: "numeric:money",
            42: "numeric:order",
            43: "numeric:other",
            44: "numeric:period",
            45: "numeric:percentage",
            46: "numeric:speed",
            47: "numeric:temperature",
            48: "numeric:size and volume",
            49: "numeric:weight"
        }
        label_list = list(id2label.values())
        pred=prediction.lower()
        for l in label_list:
            if l in prediction.lower():
                pred = l
    elif dataset_name == 'banking77':
        match = re.findall(r"[a-zA-Z]+(?:_[a-zA-Z]+)+", prediction)
        pred = match[0].strip() if match else prediction.strip()
        pred = pred.lower()
    elif dataset_name == 'clinc150':
        match = re.findall(r"([a-zA-Z_ ]+:[a-zA-Z_ ]+)", prediction)
        pred = match[0].strip() if match else prediction.strip()
        pred = pred.lower()
    elif dataset_name == 'xlsum':
        match = re.findall(r"^(.*?)(?:\n\n)", prediction, re.DOTALL)
        pred = match[0].strip() if match else prediction.strip()
    elif dataset_name == 'wmt19':
        match = re.findall(r"^(.*?)(?:\n)", prediction, re.DOTALL)
        pred = match[0].strip() if match else prediction.strip()
    elif dataset_name == 'gsm8k':
        pred = truncate_after_first_answer_gsm8k(prediction)
    elif 'math' in dataset_name:
        pred = truncate_after_first_answer_math(prediction)
    elif dataset_name in ['next_item', 'capitalize_first_letter', 'choose_first_of_5', 'english-french', 'english-german', 'park-country', 'landmark-country', 'english-spanish', 'synonym', 'country-capital', 'singular-plural','antonym']:
        match = re.findall(r"^(.*?)(?:\n)", prediction, re.DOTALL)
        pred = match[0].strip() if match else prediction.strip()
    else:
        pred = prediction
    print(prediction)
    print("_____________")
    print(pred)
    print("_____________")
    return pred

def evaluate(pred, gt, dataset_name):
    if dataset_name == 'trec' or dataset_name == 'banking77' or dataset_name == 'clinc150':
        accuracy = accuracy_score(gt, pred)
        f1 = f1_score(gt, pred, average='macro')
        return {'accuracy' : accuracy, 'f1-score' : f1}
    elif dataset_name == 'xlsum':
        rouge = Rouge()
        score = rouge.get_scores(pred, gt, avg=True)
        return score
    elif dataset_name == 'wmt19':
        score = sacrebleu.corpus_bleu(pred, [gt]).score
        return {'bleu-score' : score}
    elif dataset_name == 'gsm8k':
        pred = [extract_answer_gsm8k(p) for p in pred]
        gt = [extract_answer_gsm8k(g) for g in gt]
        score = sum(p==g for p,g in zip(pred, gt)) / len(pred)
        return {'exact_match' : score}
    elif 'math' in dataset_name:
        pred = [remove_boxed(extract_answer_math(p)) for p in pred]
        gt = [remove_boxed(extract_answer_math(g)) for g in gt]
        score = sum(is_equiv(p,g) for p,g in zip(pred,gt)) / len(pred)
        return {'exact_match' : score}
    elif dataset_name in ['next_item', 'capitalize_first_letter', 'choose_first_of_5', 'english-french', 'english-german', 'park-country', 'landmark-country', 'english-spanish', 'synonym', 'country-capital', 'singular-plural','antonym']:
        score = sum(p==g for p,g in zip(pred, gt)) / len(pred)
        return {'exact_match' : score}
        

def icl_without_intervention(train_dataset, test_dataset, n_shots, model, tokenizer,  
                             prefixes=None, separators=None, generate_str = False, dataset_name = None, max_new_tokens=500, insert_inst=True):
    res = []
    gt = []
    pred = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        stop_strings=None
        if n_shots == 0:
            demonstrations = []
        else:
            demonstrations = random.sample(train_dataset, n_shots)
            # stop_strings=['\n\n']
        test_query = test_dataset[j]['input']
        prompt = create_prompt(demonstrations, test_query, prefixes, separators, insert_inst=insert_inst)
        inputs = tokenizer(prompt, return_tensors='pt')
        test_target = test_dataset[j]['output']
        output_str = n_shot_eval(inputs, model, tokenizer, generate_str=generate_str, max_new_tokens=max_new_tokens, stop_strings=stop_strings)
        cleaned_output_str = post_process(output_str, dataset_name)
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "prediction" : output_str,
            "cleaned_prediction" : cleaned_output_str,
            "gt" : test_target,
        })
        gt.append(test_target)
        pred.append(cleaned_output_str)
        score = evaluate(pred, gt, dataset_name=dataset_name)
    return res, score


def icl_with_intervention(test_dataset, icv, model, model_config, tokenizer, 
                          prefixes=None, separators=None, eval_edit_layer=[0], add=True, generate_str = False, dataset_name=None, max_new_tokens=500, insert_inst = True):
    res = []
    gt = []
    pred = []
    for j in tqdm(range(len(test_dataset)), total = len(test_dataset)):
        test_query = test_dataset[j]['input']
        prompt = create_prompt([], test_query, prefixes, separators, insert_inst=insert_inst)
        test_target = test_dataset[j]['output']
        intervention_output = n_shot_eval_intervention(prompt, eval_edit_layer, icv, model, model_config, tokenizer, add, generate_str, max_new_tokens)
        cleaned_output_str = post_process(intervention_output, dataset_name=dataset_name)
        res.append({
            "input_prompt" : prompt,
            "test_query" : test_query,
            "prediction" : intervention_output,
            "cleaned_prediction" : cleaned_output_str,
            "gt" : test_target,
        })
        gt.append(test_target)
        pred.append(cleaned_output_str)
        score = evaluate(pred, gt, dataset_name=dataset_name)
    return res, score

def choose_repre_metric(score, dataset_name=None):
    if dataset_name == 'trec' or dataset_name == 'banking77' or dataset_name == 'clinc150':
        return score['f1-score']
    elif dataset_name == 'xlsum':
        return score['rouge-l']['f']
    elif dataset_name == 'wmt19':
        return score['bleu-score']
    elif dataset_name == 'gsm8k':
        return score['exact_match']
    elif 'math' in dataset_name:
        return score['exact_match']
    elif dataset_name in ['next_item', 'capitalize_first_letter', 'choose_first_of_5', 'english-french', 'english-german', 'park-country', 'landmark-country', 'english-spanish', 'synonym', 'country-capital', 'singular-plural', 'antonym']:
        return score['exact_match']
    else:
        return None

