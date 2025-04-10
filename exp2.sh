subset=(banking77 clinc150 trec)

for subset_name in "${subset[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 Few_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_7b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-7b-hf
    CUDA_VISIBLE_DEVICES=1 python3 Zero_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_7b --max_new_tokens 10 --model_name meta-llama/Llama-2-7b-hf
    CUDA_VISIBLE_DEVICES=1 python3 Diff_icv.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_7b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-7b-hf
done

for subset_name in "${subset[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 Task_vector.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_7b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-7b-hf
    CUDA_VISIBLE_DEVICES=1 python3 Diff_icv_baseline.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_7b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-7b-hf
done

for subset_name in "${subset[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 Few_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_13b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-13b-hf
    CUDA_VISIBLE_DEVICES=1 python3 Zero_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_13b --max_new_tokens 10 --model_name meta-llama/Llama-2-13b-hf
    CUDA_VISIBLE_DEVICES=1 python3 Diff_icv.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_13b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-13b-hf
done

for subset_name in "${subset[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 Task_vector.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_13b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-13b-hf
    CUDA_VISIBLE_DEVICES=1 python3 Diff_icv_baseline.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_llama2_13b --n_shots 100 --max_new_tokens 10 --model_name meta-llama/Llama-2-13b-hf
done