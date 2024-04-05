#! /usr/bin/env bash

set -ex
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
MAX_SOURCE_LEN=1024
MAX_TARGET_LEN=128
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=32
MAX_STEP=20
SAVE_INTERVAL=10

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=advertise_gen_pt

BASE_MODEL_PATH=../../chatglm3-6b-32k
DATASET_PATH=formatted_data/advertise_gen.jsonl
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS ../finetune.py \
    --deepspeed ../configs/deepspeed.json\
    --train_format input-output \
    --train_file $DATASET_PATH \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
