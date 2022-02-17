import sys, os
from collections import Counter

import tensorflow as tf
from tensor2tensor.data_generators import problem, text_problems
from tensor2tensor.utils import registry

tf.compat.v1.enable_eager_execution()

'''
    DATA_DIR: the data directory that contains the training and validation data
    BINARY_DATA_DIR: the directory for storing the binary data for tensor2tensor
'''

DATA_DIR = sys.argv[1]
BINARY_DATA_DIR = sys.argv[2]

TRAIN_DATA_DIR = os.path.join(DATA_DIR,'train')
VALID_DATA_DIR = os.path.join(DATA_DIR,'eval')

INPUT_FILENAME = 'before.txt'
TARGET_FILENAME = 'after.txt'

# create directory to store binary data for tensor2tensor
tf.io.gfile.makedirs(BINARY_DATA_DIR)

# register problem and set properties
@registry.register_problem
class AutoTransform(text_problems.Text2TextProblem):
    @property
    def vocab_type(self):
    # We can use different types of vocabularies, `VocabType.CHARACTER`,
    # `VocabType.SUBWORD` and `VocabType.TOKEN`.
    #
    # SUBWORD and CHARACTER are fully invertible -- but SUBWORD provides a good
    # tradeoff between CHARACTER and TOKEN.
        return text_problems.VocabType.TOKEN

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return '<UNK>'

    @property
    def is_generate_per_split(self):
        # If we have pre-existing data splits for (train, eval, test) then we set
        # this to True, which will have generate_samples be called for each of the
        # dataset_splits.
        #
        # If we do not have pre-existing data splits, we set this to False, which
        # will have generate_samples be called just once and the Problem will
        # automatically partition the data into dataset_splits.
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            file_path = TRAIN_DATA_DIR
        elif dataset_split == problem.DatasetSplit.EVAL:
            file_path = VALID_DATA_DIR

        with open(os.path.join(file_path, INPUT_FILENAME), 'r') as f:
            buggy_code = f.readlines()
        with open(os.path.join(file_path, TARGET_FILENAME), 'r') as f:
            fixed_code = f.readlines()

        # add <EOS> to indicate end of string
        buggy_code_processed = [s.replace('\n',' <EOS>').strip() for s in buggy_code]
        fixed_code_processed = [s.replace('\n',' <EOS>').strip() for s in fixed_code]

        # this code will be used by tensor2tensor to generate binary file
        # leave it as-is
        for i in range(len(buggy_code_processed)):
            yield{
                "inputs": buggy_code_processed[i],
                "targets": fixed_code_processed[i]
            }

    def generate_vocab(self, data_dir):

        with open(os.path.join(TRAIN_DATA_DIR, INPUT_FILENAME), 'r') as f:
            buggy_code = f.readlines()
        with open(os.path.join(TRAIN_DATA_DIR, TARGET_FILENAME), 'r') as f:
            fixed_code = f.readlines()

        vocab_counter = Counter()

        for line in buggy_code:
            vocab_counter.update(line.split())

        for line in fixed_code:
            vocab_counter.update(line.split())

        vocab_list = list(vocab_counter.keys())

        # leave it as-is
        # because the index of <pad> and <EOS> is specified by tensor2tensor
        # so don't change the order or remove <EOS> and <pad> tokens
        vocab_list = ["<pad>", "<EOS>"] + vocab_list + ['<UNK>']
        vocab_str = '\n'.join(vocab_list)

        with open(os.path.join(data_dir,'vocab.'+self.name+'.tokens'),'w') as f:
            f.write(vocab_str)

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 100,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 100,
        }]

my_problem = AutoTransform()

my_problem.generate_vocab(BINARY_DATA_DIR)
my_problem.generate_data(BINARY_DATA_DIR, '')