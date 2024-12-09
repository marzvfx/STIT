#!/bin/bash

# Define expressions and parameters
expressions=("angry" "contempt" "disgusted" "fear" "happy" "sad" "surprised")
neutral="neutral"
dataset_root="/mnt/lipdub/Data/train/expression/preprocessed_data"
visualization_folder="/mnt/lipdub/Data/train/expression/stylegan_inversions/"
latent_directory_root="/mnt/lipdub/Data/train/expression/stylegan_inversions/latents"
num_frames=100
samples_list_dir="/mnt/lipdub/Data/train/expression/preprocessed_data/samples_cache.pkl"  # Cache file path

# Proceed with emotion search tasks
for target_expression in "${expressions[@]}"; do
    echo "Running: ${neutral} -> ${target_expression}"
    python3 find_edit_direction.py \
        --starting_expression "$neutral" \
        --target_expression "$target_expression" \
        --dataset_root "$dataset_root" \
        --visualization_folder "$visualization_folder" \
        --latent_directory_root "$latent_directory_root" \
        --num_frames_per_identity "$num_frames" \
        --samples_list_dir "$samples_list_dir"
done

for starting_expression in "${expressions[@]}"; do
    for target_expression in "${expressions[@]}"; do
        if [ "$starting_expression" != "$target_expression" ]; then
            echo "Running: ${starting_expression} -> ${target_expression}"
            python3 find_edit_direction.py \
                --starting_expression "$starting_expression" \
                --target_expression "$target_expression" \
                --dataset_root "$dataset_root" \
                --visualization_folder "$visualization_folder" \
                --latent_directory_root "$latent_directory_root" \
                --num_frames_per_identity "$num_frames" \
                --samples_list_dir "$samples_list_dir"
        fi
    done
done

