CUDA_VISIBLE_DEVICES=0 python3 Task_vector.py --dataset_name banking77 --seed 42 --save_path_root ./results --n_shots 100 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=0 python3 Diff_icv_baseline.py --dataset_name banking77 --seed 42 --save_path_root ./results --n_shots 100 --max_new_tokens 20  

CUDA_VISIBLE_DEVICES=0 python3 Task_vector.py --dataset_name banking77 --seed 41 --save_path_root ./results --n_shots 100 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=0 python3 Diff_icv_baseline.py --dataset_name banking77 --seed 41 --save_path_root ./results --n_shots 100 --max_new_tokens 20  

CUDA_VISIBLE_DEVICES=0 python3 Task_vector.py --dataset_name banking77 --seed 40 --save_path_root ./results --n_shots 100 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=0 python3 Diff_icv_baseline.py --dataset_name banking77 --seed 40 --save_path_root ./results --n_shots 100 --max_new_tokens 20  