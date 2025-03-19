CUDA_VISIBLE_DEVICES=1 python3 Task_vector.py --dataset_name wmt19 --seed 42 --save_path_root ./results --n_shots 50 --max_new_tokens 200
CUDA_VISIBLE_DEVICES=1 python3 Diff_icv_baseline.py --dataset_name wmt19 --seed 42 --save_path_root ./results --n_shots 50 --max_new_tokens 200  

CUDA_VISIBLE_DEVICES=1 python3 Task_vector.py --dataset_name wmt19 --seed 41 --save_path_root ./results --n_shots 50 --max_new_tokens 200
CUDA_VISIBLE_DEVICES=1 python3 Diff_icv_baseline.py --dataset_name wmt19 --seed 41 --save_path_root ./results --n_shots 50 --max_new_tokens 200  

CUDA_VISIBLE_DEVICES=1 python3 Task_vector.py --dataset_name wmt19 --seed 41 --save_path_root ./results --n_shots 50 --max_new_tokens 200
CUDA_VISIBLE_DEVICES=1 python3 Diff_icv_baseline.py --dataset_name wmt19 --seed 41 --save_path_root ./results --n_shots 50 --max_new_tokens 200  


