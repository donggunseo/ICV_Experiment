subset=(banking77 clinc150 trec)

for subset_name in "${subset[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 Few_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results/no_inst --n_shots 100 --max_new_tokens 10 --insert_inst
    CUDA_VISIBLE_DEVICES=1 python3 Zero_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results/no_inst --max_new_tokens 10 --insert_inst
    CUDA_VISIBLE_DEVICES=1 python3 Diff_icv.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results/no_inst --n_shots 100 --max_new_tokens 10 --insert_inst
done

CUDA_VISIBLE_DEVICES=1 python3 Few_shot.py --dataset_name wmt19 --seed 41 --save_path_root ./results/no_inst --n_shots 50 --max_new_tokens 200 --insert_inst
CUDA_VISIBLE_DEVICES=1 python3 Zero_shot.py --dataset_name wmt19 --seed 41 --save_path_root ./results/no_inst --max_new_tokens 200 --insert_inst
CUDA_VISIBLE_DEVICES=1 python3 Diff_icv.py --dataset_name wmt19 --seed 41 --save_path_root ./results/no_inst --n_shots 50 --max_new_tokens 200 --insert_inst

for subset_name in "${subset[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 Task_vector.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results/no_inst --n_shots 100 --max_new_tokens 10 --insert_inst
    CUDA_VISIBLE_DEVICES=1 python3 Diff_icv_baseline.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results/no_inst --n_shots 100 --max_new_tokens 10 --insert_inst
done

CUDA_VISIBLE_DEVICES=1 python3 Task_vector.py --dataset_name wmt19 --seed 41 --save_path_root ./results/no_inst --n_shots 50 --max_new_tokens 200 --insert_inst
CUDA_VISIBLE_DEVICES=1 python3 Diff_icv_baseline.py --dataset_name wmt19 --seed 41 --save_path_root ./results/no_inst --n_shots 50 --max_new_tokens 200 --insert_inst