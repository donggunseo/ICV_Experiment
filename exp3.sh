CUDA_VISIBLE_DEVICES=3 python3 Few_shot.py --dataset_name clinc150 --seed 42 --save_path_root ./results --n_shots 100 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python3 Few_shot.py --dataset_name clinc150 --seed 41 --save_path_root ./results --n_shots 100 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python3 Few_shot.py --dataset_name clinc150 --seed 40 --save_path_root ./results --n_shots 100 --max_new_tokens 20

CUDA_VISIBLE_DEVICES=3 python3 Zero_shot.py --dataset_name clinc150 --seed 42 --save_path_root ./results  --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python3 Zero_shot.py --dataset_name clinc150 --seed 41 --save_path_root ./results  --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python3 Zero_shot.py --dataset_name clinc150 --seed 40 --save_path_root ./results  --max_new_tokens 20

CUDA_VISIBLE_DEVICES=3 python3 Diff_icv.py --dataset_name clinc150 --seed 42 --save_path_root ./results --n_shots 100 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python3 Diff_icv.py --dataset_name clinc150 --seed 41 --save_path_root ./results --n_shots 100 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python3 Diff_icv.py --dataset_name clinc150 --seed 40 --save_path_root ./results --n_shots 100 --max_new_tokens 20