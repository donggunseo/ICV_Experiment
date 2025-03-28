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
    parser.add_argument('--max_new_tokens', help="Number of tokens to generate", type=int, required=False, default=500)
    parser.add_argument('--insert_inst', help="whether insert task instructino or not", action="store_false")
    args = parser.parse_args()

    seed = args.seed
    dataset_name = args.dataset_name
    model_name = args.model_name
    save_path_root = f"{args.save_path_root}/{dataset_name}/{seed}/"
    device = args.device
    max_new_tokens = args.max_new_tokens
    insert_inst = args.insert_inst

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

    print("Zero-shot")
    zs_result, zs_score= icl_without_intervention(train_dataset=train_dataset, test_dataset=test_dataset, n_shots=0, model=model, tokenizer=tokenizer, 
                                                  prefixes=prefixes, separators=separators, generate_str=generate_str, dataset_name=dataset_name, max_new_tokens=max_new_tokens, insert_inst=insert_inst)
    print(f"Zero-shot ICL result: {zs_score}")
    zs_result = {'score': zs_score, 'result':zs_result}
    with open(save_path_root+"zs_result.json", 'w') as f:
        json.dump(zs_result, f)