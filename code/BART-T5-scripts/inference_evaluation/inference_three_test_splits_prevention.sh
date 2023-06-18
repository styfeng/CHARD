#!/bin/bash

export MODEL_TYPE=$1
export MODEL_SIZE=$2
export SEED=$3
export ENCODER_LEN=$4
export DECODER_LEN=$5
export MODEL_NAME=$6
export LR=$7
export CHECKPOINT=$8
export BATCH_SIZE=$9
export GPU=${10}

FILES=("HF_training_data/prevention/prevention_test_HF.json" "HF_training_data/prevention/prevention_test-seen_HF.json" "HF_training_data/prevention/prevention_test-unseen_HF.json")

for TEST_FILE in "${FILES[@]}"
do
	echo "Inference on test file: "$TEST_FILE
	echo "Model type: "$MODEL_TYPE
	echo "Model size: "$MODEL_SIZE
	echo "Seed: "$SEED
	echo "Encoder len: "$ENCODER_LEN
	echo "Decoder len: "$DECODER_LEN
	echo "Model name: "$MODEL_NAME
	echo "Learning rate: "$LR
	echo "Checkpoint: "$CHECKPOINT
	echo "Batch size: "$BATCH_SIZE
	echo "GPU: "$GPU
	python BART-T5-scripts/${MODEL_TYPE}_inference_evaluate.py ${MODEL_SIZE} ${SEED} ${ENCODER_LEN} ${DECODER_LEN} ${MODEL_NAME} $TEST_FILE ${LR} ${CHECKPOINT} ${BATCH_SIZE} ${GPU}
done