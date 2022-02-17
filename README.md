
# Supplementary Materials for "AutoTransform: Automated Code Transformation to Support Modern Code Review Process"

Table of Content:

- Experiment
- A Replication Package

## Experiment

### Experimental Setup

In this work, we set a similar number of epochs for each model for each dataset. The number of epochs indicates how many times that the whole training data was used to train the model. The number of epoch is based on the size of trainging data, batch size, and the number of train steps ([see](https://deeplizard.com/learn/video/U4WB9p6ODjM)). However, the calculation of epochs for each NMT library (i.e., Tensor2Tensor and Seq2Seq) is [different](https://github.com/tensorflow/tensor2tensor/issues/415).

- #Epoch for T2T = #train_steps * batch_size / **#subwords** ([see](https://arxiv.org/abs/1804.00247)) (Since in T2T, batch_size is the number of subwords per batch)

- #Epoch for Seq2Seq = #train_steps * batch_size / **#sequences** ([see](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)) (Since in Seq2Seq, batch_size is the number of sequences per batch)


where #train_steps and batch_size are the hyper-parameters used in the NMT models.

-  **epoch_calculation.pdf** provides the number of #epoch, #train_steps, and batch size used for each dataset.

  
### Datasets

The datasets include 147,553 changed methods that were extracted from three Gerrit code review repositories, namely Android, Google, and Ovirt.

Each changed method has a pair of the **before** and **after** versions.

There are a total of 12 datasets that were used in our experiments, i.e., 3 repositories * 2 method sizes (small and medium) * 2 change types (with and without new tokens appearing in the after version).

Each dataset was partitioned into training (80%), validation (10%), and testing (10%). 
- Subword-tokenized code (subword_tokenized.tgz)
- Original source code (orginal_code.tgz)

The datasets are publicly available at Zenodo [Link](https://zenodo.org/record/6068004).


### Predictions

For each dataset, we provide the generated sequences using beam width = {1, 5, 10}. Both raw generation (i.e., subword sequences or abstracted code sequeces) and code generation (converted to code sequences) are provided. 

- RQ1: Predictions of our AutoTransform (BPE2K+Transformer) and Tufano et al approach (Abs+RNN)
- RQ2: Predictions of the additional four combinations (i.e., BPE5K+Transformer, BPE2K+RNN, BPE5K+RNN, Abs+Transformer)

The predictions are publicly available at Zenodo [Link](https://zenodo.org/record/6068004).

## A Replication Package

We provide scripts for our Autotransform. We used the [subword-nmt](https://github.com/rsennrich/subword-nmt) library for subword tokenization and used the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library for training and inference.

  
### Scripts

The ``scripts/`` folder contains the following scripts:

 - ``subword_tokenize.sh``: script that performs subword tokenize on the original code
 - ``generate_binary_data.py``: script that converts input data into a binary object for T2T when training the model.
 - ``train_model.sh``: script that performs training
 - ``inference.sh``: script that performs inference
 - ``AutoTransform/AutoTransform_problem.py``: is for specifying hyper-parameter settings  

### Usage

Before using our scripts, make sure to have a working installation of [subword-nmt](https://github.com/rsennrich/subword-nmt) and [tensor2tensor](https://github.com/tensorflow/tensor2tensor). Next, download our scripts and run the script in the folder ``scripts``

#### (Step 0) Data Structure
For each dataset, it should be structured as: ``<data_dir>/<partition>/<version>.txt``

where

 - ``<data_dir>``: the directory path to the dataset of the original code of changed methods
 - ``<partition>``: one of {train, eval, test}
 - ``<version>``: one of {before, after}
 
 Example:
 
 ```
 dataset/without_new_tokens/android/small/train/before.txt
 dataset/without_new_tokens/android/small/train/after.txt
 ```

#### (Step 1) Subword tokenize using BPE
This script will perform subword tokenize for training, validation, and testing data.

```
bash subword_tokenize.sh <data_dir> <tokenized_data_dir> <num_merges>
```

- ``<data_dir>``: the directory path to the dataset of the original code of changed methods
- ``<tokenized_data_dir>``: the directory for storing the subword tokenized data with the same structure as above.
- ``<num_merges>``: The number of merge operations

Example:

```
bash subword_tokenize.sh dataset/without_new_tokens/android/small/ subword_tokenized/without_new_tokens/android/small/ 2000
```

#### (Step 2) Generate binary data for T2T
This script will convert the text files of the before and after versions in the training and validation data into binary files.

```
bash generate_binary_data.py <data_dir> <binary_data_dir>
```

- ``<data_dir>``: the data directory that contains the training and validation data
- ``<binary_data_dir>``: the directory for storing the binary data for tensor2tensor

Example:

```		
python generate_binary_data.py subword_tokenized/without_new_tokens/android/small/ binary_data/without_new_tokens/android/small/
```

#### (Step 3) Model Training
This script will train the Transformer model based on the specified hyper-parameter setting and the number of train steps (See *epoch_calculation.pdf* about how we calculate the number of train steps)

```
bash train_model.sh <binary_data_dir> <hyper-parameter_setting> <model_dir> <train_step>
```

- ``<binary_data_dir>``: the directory that stores the binary data generated in Step 2
- ``<hyper-parameter_setting>: the setting name of the hyper-parameters defined in ``AutoTransform/AutoTransform_problem.py``
- ``<model_dir>``: the directory for saving the model
- ``<train_step>``: the number of train steps to train model

Example:

```
bash train_model.sh binary_data/without_new_tokens/android/small/ transformer_hparams1 models/without_new_tokens/android/small/setting1 2000
```

#### (Step 4) Inference
This script will generate a prediction for each method in the input_file. The output will be saved under the model directory (i.e., ``<model_dir>/predictions``)

```
bash inference.sh <binary_data_dir> <model_dir> <input_file> <hyper-parameter_setting> <ckpt_number> <beam_width>
```

- ``<binary_data_dir>``: the directory that stores the binary data generated in Step 2
- ``<model_dir>``: the directory that contains the model
- ``<input_file>``: the text file of the before version in the testing data
- ``<hyper-parameter_setting>``: the setting name of the hyper-parameters defined in ``AutoTransform/AutoTransform_problem.py``
- ``<ckpt_number>``: the checkpoint number (i.e., the number of train steps that the model was used to train). For example, ``ckpt_number=1000`` means that  we will use the model that was trained using 1000 train steps.
- ``<beam_width>``: the number of generated sequences for each input instance (i.e., a method)

Example:

```
bash inference.sh binary_data/without_new_tokens/android/small/ models/without_new_tokens/android/small/setting1 subword_tokenized/without_new_tokens/android/small/test/before.txt transformer_hparams1 2000 5
```
