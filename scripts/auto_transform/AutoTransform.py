from tensor2tensor.data_generators import problem, text_problems
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

# If you want to create your own problem for tensor2tensor, follow the below steps
# 1. identify problem name (here the problem name is AutoTransform), note that the problem name should be camel-case (split word by capital letters)
# 2. create directory with the problem name but use snake case (such as auto_transform)
# 3. create python file in directory of step 2. (see this file for more detail)

@registry.register_problem
class AutoTransform(text_problems.Text2TextProblem):
    @property
    def vocab_type(self):
    # We can use different types of vocabularies, `VocabType.CHARACTER`,
    # `VocabType.SUBWORD` and `VocabType.TOKEN`.

        return text_problems.VocabType.TOKEN

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return '<UNK>'

# note: 
#     the hyper-parameter setting must be camel-case (see example below).
#     when using these hyper-parameter settings in tensor2tensor, each word that begins with capital letter will be split by '_', and all words will be converted to lower-case (for example, TransformerHparams1 -> transformer_hparams1)

@registry.register_hparams
def TransformerHparams1():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    hparams.num_encoder_layers = 1
    hparams.num_decoder_layers = 2
    hparams.hidden_size = 256
    hparams.num_heads = 8
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200
    return hparams
    
@registry.register_hparams
def TransformerHparams2():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    hparams.num_encoder_layers = 2
    hparams.num_decoder_layers = 4
    hparams.hidden_size = 256
    hparams.num_heads = 8
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200
    return hparams

@registry.register_hparams
def TransformerHparams3():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    hparams.num_encoder_layers = 1
    hparams.num_decoder_layers = 2
    hparams.hidden_size = 512
    hparams.num_heads = 8
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200
    return hparams
    
@registry.register_hparams
def TransformerHparams4():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    hparams.num_encoder_layers = 2
    hparams.num_decoder_layers = 4
    hparams.hidden_size = 512
    hparams.num_heads = 8
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200
    return hparams

@registry.register_hparams
def TransformerHparams5():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    hparams.num_encoder_layers = 1
    hparams.num_decoder_layers = 2
    hparams.hidden_size = 256
    hparams.num_heads = 16
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200
    return hparams
    
@registry.register_hparams
def TransformerHparams6():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    # hparams.batch_size = 3000
    hparams.num_encoder_layers = 2
    hparams.num_decoder_layers = 4
    hparams.hidden_size = 256
    hparams.num_heads = 16
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200
    return hparams

@registry.register_hparams
def TransformerHparams7():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    hparams.num_encoder_layers = 1
    hparams.num_decoder_layers = 2
    hparams.hidden_size = 512
    hparams.num_heads = 16
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200

    return hparams
    
@registry.register_hparams
def TransformerHparams8():
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 5000
    hparams.num_encoder_layers = 2
    hparams.num_decoder_layers = 4
    hparams.hidden_size = 512
    hparams.num_heads = 16
    hparams.eval_drop_long_sequences = True
    hparams.max_length = 1200

    return hparams
