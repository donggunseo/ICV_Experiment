import os, json
import torch
import argparse
from utils.model_utils import *
from utils.data_utils import *
from utils.inference_utils import *
from utils.extract_utils import *
from utils.prompt_utils import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=100)
    parser.add_argument('--max_new_tokens', help="Number of tokens to generate", type=int, required=False, default=500)
    parser.add_argument('--load_icv', help="load already computed icv", action="store_true")
    parser.add_argument('--insert_inst', help="whether insert task instructino or not", action="store_false")
    args = parser.parse_args()

    seed = args.seed
    dataset_name = args.dataset_name
    model_name = args.model_name
    n_shots = args.n_shots
    save_path_root = f"{args.save_path_root}/{dataset_name}/{seed}/{n_shots}shots/"
    device = args.device
    max_new_tokens = args.max_new_tokens
    insert_inst = args.insert_inst

    prefixes = PREFIX_DICT[dataset_name]
    separators = {"input":"\n", "output":"\n\n", "instructions":"\n"}

    generate_str = is_generate_sampling(dataset_name)


    load_icv = args.load_icv and os.path.exists(save_path_root+'tack_vector.pt')
    
    print(args)
    set_seed(seed)
    # Load Model & Tokenizer
    print("Loading Model")
    model, tokenizer, model_config = load_model_and_tokenizer(model_name, device=device)
    model.eval()
    torch.set_grad_enabled(False)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root, exist_ok=True)

    print("Loading Dataset")
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_name)

    EDIT_LAYER = list(range(model_config['n_layers']))
    
    print("Task-Vector")
    if load_icv:
        icv = torch.load(save_path_root+'task_vector.pt')
    else:
        icv = get_mean_hidden_states(train_dataset=train_dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots, N_TRIALS=len(test_dataset), prefixes=prefixes, separators=separators, insert_inst=insert_inst)
        torch.save(icv, save_path_root+'task_vector.pt')
    val_score_per_layer = {l:0 for l in EDIT_LAYER}
    for l in tqdm(EDIT_LAYER):
        edit_layer = [l]
        _, task_vector_val_score = icl_with_intervention(test_dataset=val_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                       prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer, add=False, generate_str=generate_str, dataset_name=dataset_name, max_new_tokens=max_new_tokens, insert_inst=insert_inst)
        val_score_per_layer[l] = choose_repre_metric(task_vector_val_score, dataset_name)
    edit_layer = [max(val_score_per_layer, key=val_score_per_layer.get)]
    task_vector_res, task_vector_score = icl_with_intervention(test_dataset=test_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                             prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer, add=False, generate_str=generate_str, dataset_name=dataset_name, max_new_tokens=max_new_tokens, insert_inst=insert_inst)
    print(f"Task-Vector result: {task_vector_score} with edit layer {edit_layer[0]}")
    task_vector_val_score_per_layer = val_score_per_layer
    task_vector_res = {'score': task_vector_score, 'intervention_layer' : edit_layer[0], 'result':task_vector_res, 'val_f1_per_layer':task_vector_val_score_per_layer}
    with open(save_path_root+"task_vector_result.json", "w") as f:
        json.dump(task_vector_res, f)