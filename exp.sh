# subset=(antonym capitalize_first_letter choose_first_of_5 country-capital english-french english-german english-spanish park-country landmark-country singular-plural synonym )
subset=(country-capital park-country)
for subset_name in "${subset[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 Few_shot.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results/synthetic_tasks --n_shots 5 --max_new_tokens 10 --insert_inst
    CUDA_VISIBLE_DEVICES=0 python3 Zero_shot.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results/synthetic_tasks --max_new_tokens 10 --insert_inst
    CUDA_VISIBLE_DEVICES=0 python3 Task_vector.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results/synthetic_tasks --n_shots 5 --max_new_tokens 10 --insert_inst
    CUDA_VISIBLE_DEVICES=0 python3 Diff_icv_baseline.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results/synthetic_tasks --n_shots 5 --max_new_tokens 10 --insert_inst
    CUDA_VISIBLE_DEVICES=0 python3 Diff_icv.py --dataset_name "$subset_name" --seed 42 --save_path_root ./results/synthetic_tasks --n_shots 5 --max_new_tokens 10 --insert_inst
done

