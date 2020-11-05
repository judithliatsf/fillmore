import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops, lookup_ops, array_ops
from tensorflow_text.python.ops import bert_tokenizer

class BertSingleSentenceFeaturizer(tf.Module):
    """Serializable BertSentenceFeaturizer that takes tensor of string as input and
    output the features required for downstream BERT layer
    """

    def __init__(self, vocab_file, num_oov=1, lower_case=False, max_sen_len=64, 
                 pad_id=0, cls_id=101, sep_id=102):
        self.vocab_file = tf.saved_model.Asset(vocab_file)
        self.vocab = self.load_vocab(vocab_file)
        self.vocab_table = self.create_table(self.vocab, num_oov=num_oov)
        self.tokenizer = bert_tokenizer.BertTokenizer(
            self.vocab_table,
            token_out_type=dtypes.int64,
            lower_case=lower_case)
        self.max_sen_len = max_sen_len
        self.cls_token_id = tf.constant(cls_id, dtype=tf.int32)
        self.sep_token_id = tf.constant(sep_id, dtype=tf.int32)
        self.pad_token_id = tf.constant(pad_id, dtype=tf.int32)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def __call__(self, text_input):
        """transform a collection of text to features required by bert
        * output names have to match downstream model input names*
        * this function only accepts ops in tf graph mode not eager mode*
        Args:
            text_input: A list of untokenized UTF-8 strings.
        Returns:
            A dictionary of `tf.Tensor`
        """
        def rag2tensor(ragged):
            """transform a 3D ragged tensor into a 2D tf tensor
            Args:
                ragged: A `RaggedTensor` of list of integers
            Returns:
                A `Tensor` of `tf.int32`
            """
            return tf.cast(tf.squeeze(ragged, -1), dtype=tf.int32).to_tensor()
        # convert text into token ids
        token_ids = self.tokenizer.tokenize(text_input)

        # flatten the ragged tensors
        token_ids = tf.cast(token_ids.merge_dims(1, 2), tf.int32)
        batch_size = token_ids.shape[0]

        # truncate or padding the tensors
        token_ids = token_ids.to_tensor(shape=[batch_size, self.max_sen_len], default_value=self.pad_token_id)

        # Add start and end token ids to token_ids
        start_tokens = tf.fill([batch_size, 1], self.cls_token_id)
        end_tokens = tf.fill([batch_size, 1], self.sep_token_id)
        token_ids = tf.concat([start_tokens, token_ids, end_tokens], axis=1)

        attention_mask = tf.cast(tf.math.not_equal(token_ids, self.pad_token_id), tf.int32)
        segment_ids = tf.zeros_like(token_ids)
        return {'input_ids': token_ids,
                'attention_mask': attention_mask,
                'token_type_ids': segment_ids}

    def create_table(self, vocab, num_oov=1):
        """Create table from vocab to look up index"""
        init = lookup_ops.KeyValueTensorInitializer(
            vocab,
            math_ops.range(
                array_ops.size(vocab, out_type=dtypes.int64), dtype=dtypes.int64),
            key_dtype=dtypes.string,
            value_dtype=dtypes.int64)
        return lookup_ops.StaticVocabularyTableV1(
            init, num_oov, lookup_key_dtype=dtypes.string)

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a list."""
        vocab = []
        with tf.io.gfile.GFile(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab.append(token)
        return vocab