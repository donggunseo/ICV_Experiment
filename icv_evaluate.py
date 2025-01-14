import os, json
import torch, numpy as np
import argparse
from utils.model_utils import *
from utils.data_utils import *
from utils.inference_utils import *
from utils.extract_utils import *
from tqdm import tqdm

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=-1) # 
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=100)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":"\n"})    
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    seed = args.seed
    device = args.device
    eval_edit_layer = args.edit_layer

    n_shots = args.n_shots

    prefixes = args.prefixes 
    separators = args.separators


    print(args)

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_model_and_tokenizer(model_name, device=device)
    model.eval()

    if args.edit_layer == -1: # sweep over all layers if edit_layer=-1
        eval_edit_layer = list(range(0, model_config['n_layers']))

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root, exist_ok=True)

    print("Loading Dataset")
    set_seed(seed)
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_name)
    val_dataset = val_dataset[:10]
    test_dataset = test_dataset[:10]


    # print(f"Vanilla {n_shots}ICL")

    # result, vanilla_icl_accuracy = icl_without_intervention(train_dataset, test_dataset, n_shots, model, model_config, tokenizer, prefixes=prefixes, separators=separators)
    # vanilla_icl = {"accuracy": vanilla_icl_accuracy, "result" : result}

    # with open(os.path.join(save_path_root, f"{dataset_name}_{n_shots}.json"), "w") as f:
    #     json.dump(vanilla_icl, f)

    print("Extract ICV")
    icv = get_attn_diff_hidden_states(train_dataset, model, model_config, tokenizer, n_icl_examples=10, N_TRIALS=10, prefixes = prefixes, separators=separators)
    print(icv.shape)

    # if isinstance(eval_edit_layer,int)==False:
    #     sample_res = []
    #     print("find the best intervention layer by validation set")
    #     dev_score_by_layer = {layer:0.0 for layer in eval_edit_layer}
    #     for layer in tqdm(eval_edit_layer):
    #         icv_by_layer = icv[layer]
    #         res, _, intervention_f1 = icl_with_intervention(val_dataset, icv_by_layer, model, model_config, tokenizer, layer, prefixes, separators)
    #         dev_score_by_layer[layer]=intervention_f1
    #     best_intervention_layer = max(dev_score_by_layer, key=dev_score_by_layer.get)
    # print(dev_score_by_layer)
    # print(best_intervention_layer)

    # res, zs_f1, intervention_f1 = icl_with_intervention(test_dataset, icv, model, model_config, tokenizer, 13, prefixes, separators)
    # with open("./sample.json", "w") as f:
    #     json.dump(res,f)











    
