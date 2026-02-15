#!/bin/bash

# Cross-step sampling sweep script
# Runs sampling with different num-sampling-steps values (120-250, interval 10)

CKPT="EqM-XL:2.pt"
MODEL="EqM-XL/2"

for STEPS in $(seq 120 10 250); do
    echo "========================================"
    echo "Running with num-sampling-steps=${STEPS}"
    echo "========================================"

    OUTPUT_FOLDER="samples_cross_step_${STEPS}"

    CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 sample_gd.py \
        --model ${MODEL} \
        --ckpt ${CKPT} \
        --cross-step \
        --num-sampling-steps ${STEPS} \
        --folder ${OUTPUT_FOLDER}

    echo "Completed: ${STEPS} steps -> ${OUTPUT_FOLDER}"
    echo ""
done

echo "All runs completed!"
