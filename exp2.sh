subset=(math_algebra math_counting_and_probability math_geometry math_intermediate_algebra math_number_theory math_prealgebra math_precalculus)

for subset_name in "${subset[@]}"; do
    echo "Processing dataset: $subset_name"
    CUDA_VISIBLE_DEVICES=1 python3 icv_evaluate.py --dataset_name "$subset_name" --seed 41 --save_path_root ./results_bf16 --n_shots 5
done

echo "All datasets processed successfully!"