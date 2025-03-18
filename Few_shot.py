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

    print(f"Vanilla {n_shots}shots-ICL")
    fs_result, fs_score= icl_without_intervention(train_dataset=train_dataset, test_dataset=test_dataset, n_shots=n_shots, model=model, tokenizer=tokenizer, 
                                                  prefixes=prefixes, separators=separators, generate_str=generate_str, dataset_name=dataset_name, max_new_tokens=max_new_tokens)
    print(f"Vanilla ICL result: {fs_score}")
    fs_result = {'score': fs_score, 'result':fs_result}
    with open(save_path_root+"fs_result.json", 'w') as f:
        json.dump(fs_result, f)