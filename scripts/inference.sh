# BINARY_T2T_DATA_DIR: directory that stores binary data for tensor2tensor
# SAVED_MODEL_DIR: directory to save model
# INPUT_FILE: before version source code (test data)
# HPARAMS_NAME: hyper-parameter name from AutoTrasform.py (for example, if the hyper-parameter setting in AutoTransform is HparamSet1, the HPARAMS_NAME here is hparam_set1)
# TRAIN_STEP: number of train step to train model 
# BEAM_SIZE: the number of possible after version source code generated for each before version (the more, the slower)

BINARY_T2T_DATA_DIR=$1
SAVED_MODEL_DIR=$2
INPUT_FILE=$3
HPARAMS_NAME=$4
TRAIN_STEP=$5
BEAM_SIZE=$6

PROBLEM=auto_transform
MODEL=transformer
USR_DIR=./auto_transform
WORKER_GPU_FRAC=0.95

PREDICTION_DIR=$SAVED_MODEL_DIR/prediction

mkdir -p $PREDICTION_DIR

DECODE_TO_FILE=$PREDICTION_DIR/prediction_beam_$BEAM_SIZE.txt

CKPT_PATH=$SAVED_MODEL_DIR/model.ckpt-$TRAIN_STEP

t2t-decoder \
    --data_dir=$BINARY_T2T_DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS_NAME \
    --output_dir=$SAVED_MODEL_DIR \
    --checkpoint_path=$CKPT_PATH \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=1.0,return_beams=True,max_input_size=1200" \
    --decode_from_file=$INPUT_FILE \
    --decode_to_file=$DECODE_TO_FILE \
    --t2t_usr_dir=$USR_DIR
    --worker_gpu_memory_fraction=$WORKER_GPU_FRAC
