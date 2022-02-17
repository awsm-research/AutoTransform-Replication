#!/bin/bash

# DATA_DIR: the directory path to the dataset of the original code of changed methods
# OUTPUT_DIR: the directory for storing the subword tokenized data with the same structure as above.
# NUM_MERGE: the number of merge operations to generate and apply

DATA_DIR=$1
OUTPUT_DIR=$2
NUM_MERGE=$3

#Setup environment variables
TRAIN_BEFORE=$DATA_DIR"/train/before.txt"
TRAIN_AFTER=$DATA_DIR"/train/after.txt"


mkdir -p $OUTPUT_DIR

#STEP1: Generate merge operations based on the training dataset
subword-nmt learn-joint-bpe-and-vocab --input $TRAIN_BEFORE $TRAIN_AFTER -s $NUM_MERGE -o $OUTPUT_DIR"/train.all.codes" --write-vocabulary $OUTPUT_DIR"/train.code_before.vocab" $OUTPUT_DIR"/train.code_bafter.vocab" --num-workers -1

#STEP2: Apply merge operations generated from learn bpe to the train, eval, test and before and after versions. Note that Java keywords listd in Java_keywords.txt will not be subtokenized.
java_keywords=$(cat ./Java_keywords.txt)

mkdir -p $OUTPUT_DIR"/train"
mkdir -p $OUTPUT_DIR"/eval"
mkdir -p $OUTPUT_DIR"/test"
files=("train/before.txt" "train/after.txt" "eval/before.txt" "eval/after.txt" "test/before.txt")
for file in ${files[*]}
do
    subword-nmt apply_bpe -c $OUTPUT_DIR"/train.all.codes" --merges $NUM_MERGE -i $DATA_DATA"/$file" -o $OUTPUT_DIR"/$file" --num-workers -1 --glossaries $java_keywords
done
