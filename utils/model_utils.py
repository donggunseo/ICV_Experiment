import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import random
from typing import *



def load_model_and_tokenizer(model_name:str, device='cuda'):
    assert model_name is not None
    model_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

    MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                    "n_layers":model.config.num_hidden_layers,
                    "resid_dim":model.config.hidden_size,
                    "name_or_path":model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                    }
    # else:
    #     raise NotImplementedError("Still working to get this model available!")

    
    return model, tokenizer, MODEL_CONFIG

def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)