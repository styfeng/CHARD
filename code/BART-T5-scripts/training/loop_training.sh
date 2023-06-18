#!/bin/bash

export MODEL_TYPE=$1
export MODEL_SIZE=$2
export SEED=$3
export ENCODER_LEN=$4
export DECODER_LEN=$5
export MODEL_NAME=$6
export TRAIN_FILE=$7
export VAL_FILE=$8
export BATCH_SIZE=$9
export EPOCHS=${10}
export GPU=${11}

read -p "Enter learning rates separated by 'space': " input

for lr in ${input[@]}
do
    echo "Training using learning rate: "$lr
	echo "Model type: "$MODEL_TYPE
	echo "Model size: "$MODEL_SIZE
	echo "Seed: "$SEED
	echo "Encoder len: "$ENCODER_LEN
	echo "Decoder len: "$DECODER_LEN
	echo "Model name: "$MODEL_NAME
	echo "Train file: "$TRAIN_FILE
	echo "Val file: "$VAL_FILE
	echo "Batch size: "$BATCH_SIZE
	echo "Epochs: "$EPOCHS
	echo "GPU: "$GPU
	python BART-T5-scripts/${MODEL_TYPE}_training.py ${MODEL_SIZE} $lr ${SEED} ${ENCODER_LEN} ${DECODER_LEN} ${MODEL_NAME} ${TRAIN_FILE} ${VAL_FILE} ${BATCH_SIZE} ${EPOCHS} ${GPU}
done