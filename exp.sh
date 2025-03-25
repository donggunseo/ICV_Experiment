subset=(next_item capitalize_first_letter choose_first_of_5 english-french english-german park-country landmark-country)
for subset_name in "${subset[@]}"; do
    # CUDA_VISIBLE_DEVICES=3 python3 Few_shot.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Zero_shot.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Task_vector.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    CUDA_VISIBLE_DEVICES=3 python3 Diff_icv_baseline.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Diff_icv.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results --n_shots 5 --max_new_tokens 20

    # CUDA_VISIBLE_DEVICES=3 python3 Few_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Zero_shot.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Task_vector.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    CUDA_VISIBLE_DEVICES=3 python3 Diff_icv_baseline.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Diff_icv.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results --n_shots 5 --max_new_tokens 20

    # CUDA_VISIBLE_DEVICES=3 python3 Few_shot.py --dataset_name "$subset_name" --seed 40 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Zero_shot.py --dataset_name "$subset_name" --seed 40 --save_path_root ./results --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Task_vector.py --dataset_name "$subset_name" --seed 40 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    CUDA_VISIBLE_DEVICES=3 python3 Diff_icv_baseline.py --dataset_name "$subset_name" --seed 40 --save_path_root ./results --n_shots 5 --max_new_tokens 20
    # CUDA_VISIBLE_DEVICES=3 python3 Diff_icv.py --dataset_name "$subset_name" --seed 40 --save_path_root ./results --n_shots 5 --max_new_tokens 20
done

CUDA_VISIBLE_DEVICES=3 python3 Diff_icv_baseline.py --dataset_name english-spanish --seed 42 --save_path_root ./results --n_shots 5 --max_new_tokens 20