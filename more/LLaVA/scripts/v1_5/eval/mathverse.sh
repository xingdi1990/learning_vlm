#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_path="checkpoints/llava-v1.5-13b"
answers_file="./playground/data/eval/mathverse/outputs/pred_multimath-13b-llava-v1_5.json"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./playground/data/eval/mathverse/infer.py \
        --model_path checkpoints/llava-v1.5-13b \
        --prompt none \
        --conv_mode vicuna_v1 \
        --question_file ./playground/MathVerse/testmini.json \
        --image_folder ./playground/MathVerse/images \
        --answers_file ${answers_file}\
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

python ./playground/data/eval/mathverse/merge_pred.py --answers_file ${answers_file}

# answers_file="./eval_mathverse/outputs/pred_multimath-7b-llava-v1_5.json"

python ./playground/data/eval/mathverse/evaluate.py \
    --answers_file ${answers_file} \
    --gt_file ./playground/MathVerse/testmini.json
