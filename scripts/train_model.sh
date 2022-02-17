# BINARY_T2T_DATA_DIR: directory that stores binary data for tensor2tensor
# HPARAMS_NAME: hyper-parameter name from AutoTrasform.py (for example, if the hyper-parameter setting in AutoTransform is HparamSet1, the HPARAMS_NAME here is hparam_set1)
# SAVED_MODEL_DIR: directory to save model
# TRAIN_STEP: number of train step to train model 

# note: you can add your own hyper-parameter setting in AutoTransform.py

BINARY_T2T_DATA_DIR=$1
HPARAMS_NAME=$2           
SAVED_MODEL_DIR=$3        
TRAIN_STEP=$4    

# define constant

PROBLEM=auto_transform
MODEL=transformer
EVAL_STEPS=100
SCHEDULE=continuous_train_and_eval
USR_DIR=./auto_transform
MAX_MODEL_TO_KEEP=30

mkdir -p $SAVED_MODEL_DIR

t2t-trainer \
    --data_dir=$BINARY_T2T_DATA_DIR \
    --problem=$PROBLEM \
    --eval_throttle_seconds=10 \
    --model=$MODEL \
    --hparams_set=$HPARAMS_NAME \
    --keep_checkpoint_max=$MAX_MODEL_TO_KEEP \
    --schedule=$SCHEDULE \
    --output_dir=$SAVED_MODEL_DIR \
    --train_steps=$TRAIN_STEP \
    --eval_steps=$EVAL_STEPS \
    --t2t_usr_dir=$USR_DIR \
    --random_seed=0
