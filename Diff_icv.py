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
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results_fp16')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=100)
    parser.add_argument('--max_new_tokens', help="Number of tokens to generate", type=int, required=False, default=500)
    parser.add_argument('--load_icv', help="load already computed icv", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    dataset_name = args.dataset_name
    model_name = args.model_name
    n_shots = args.n_shots
    save_path_root = f"{args.save_path_root}/{dataset_name}/{seed}/{n_shots}shots/"
    device = args.device
    max_new_tokens = args.max_new_tokens

    prefixes = PREFIX_DICT[dataset_name]
    separators = {"input":"\n", "output":"\n\n", "instructions":"\n"}

    generate_str = is_generate_sampling(dataset_name)


    load_icv = args.load_icv and os.path.exists(save_path_root+'stacked_diff_icv.pt')
    
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


    print("Diff-ICV Stacked")
    if load_icv:
        icv = torch.load(save_path_root+'stacked_diff_icv.pt')
    else:
        icv= get_diff_stacked_hidden_states(train_dataset=train_dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots, N_TRIALS=len(test_dataset), prefixes=prefixes, separators=separators)
        torch.save(icv, save_path_root+'stacked_diff_icv.pt')
    edit_layer = list(range(model_config['n_layers']))
    stacked_diff_icv_res, stacked_diff_icv_score = icl_with_intervention(test_dataset=test_dataset, icv=icv, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                         prefixes=prefixes, separators=separators, eval_edit_layer=edit_layer, add = True, generate_str=generate_str, dataset_name=dataset_name, max_new_tokens = max_new_tokens)
    print(f"Diff-ICV Stacked result : {stacked_diff_icv_score}")
    stacked_diff_icv_res = {'score': stacked_diff_icv_score, 'result': stacked_diff_icv_res}
    with open(save_path_root+"stacked_diff_icv_result.json", "w") as f:
        json.dump(stacked_diff_icv_res, f)

    













    
