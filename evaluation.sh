#!/bin/bash

MODEL_DIR="latentqa/out/model/"
TINY_TASKS="tinyMMLU,tinyHellaswag,tinyArc,tinyTruthfulQA,tinyWinogrande,tinyGSM8k"
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
BENCHMARKS="mmlu"
BATCH_SIZE=8

WRITE_TO_FILE="out"

# lm_eval --model hf \
#     --model_args pretrained=$BASE_MODEL,peft="${MODEL_DIR}${MODEL_NAME}"\
#     --tasks $TINY_TASKS \
#     --device cuda:0 \
#     --batch_size $BATCH_SIZE \
#     --output_path $WRITE_TO_FILE 

# lm_eval --model hf \
#     --model_args pretrained=$BASE_MODEL\
#     --tasks $TINY_TASKS \
#     --device cuda:0 \
#     --batch_size $BATCH_SIZE \
#     --output_path $WRITE_TO_FILE 

lm_eval --model hf \
    --model_args pretrained=$BASE_MODEL\
    --tasks gsm8k\
    --device cuda:0 \
    --batch_size $BATCH_SIZE \
    --output_path $WRITE_TO_FILE 

MODEL_NAME="steer_promote_veganism_dolly_10"
lm_eval --model hf \
    --model_args pretrained=$BASE_MODEL,peft="${MODEL_DIR}${MODEL_NAME}"\
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size $BATCH_SIZE \
    --output_path $WRITE_TO_FILE 

MODEL_NAME="steer_promote_veganism_dolly_20"
lm_eval --model hf \
    --model_args pretrained=$BASE_MODEL,peft="${MODEL_DIR}${MODEL_NAME}"\
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size $BATCH_SIZE \
    --output_path $WRITE_TO_FILE 

MODEL_NAME="steer_promote_veganism_dolly_30"
lm_eval --model hf \
    --model_args pretrained=$BASE_MODEL,peft="${MODEL_DIR}${MODEL_NAME}"\
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size $BATCH_SIZE \
    --output_path $WRITE_TO_FILE 