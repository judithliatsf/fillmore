import tensorflow as tf
from fillmore.model import BertSingleSentenceFeaturizer
from transformers import *
import os

class BertSingleSentenceFeaturizerTest(tf.test.TestCase):
    def setUp(self):
        self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/vocab.txt")
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased")

    def test_bert_featurizer(self):
        tokenizer = BertSingleSentenceFeaturizer(
            self.vocab_path,
            max_sen_len=7
        )
        
        input_text = [
            'mary hopped.',
            'jane skipped.',
            'who farted?',
            'the quick brown fox jumped over the lazy dogs.'
        ]
        input_tensor = tf.constant(input_text)

        output = tokenizer(input_tensor)
        print(output)
        expected_token_ids = tf.constant([[101,  2984, 17230,  1012,     0,     0,     0,     0,   102],
                                          [101,  4869, 16791,  1012,     0,     0,     0,     0,   102],
                                          [101,  2040,  2521,  3064,  1029,     0,     0,     0,   102],
                                          [101,  1996,  4248,  2829,  4419,  5598,  2058,  1996,   102]],
                                         dtype=tf.int32)

        expected_attention_mask = tf.constant([[1, 1, 1, 1, 0, 0, 0, 0, 1],
                                               [1, 1, 1, 1, 0, 0, 0, 0, 1],
                                               [1, 1, 1, 1, 1, 0, 0, 0, 1],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=tf.int32

        )
        expected_token_type_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32
        )
        self.assertAllEqual(output['input_word_ids'], expected_token_ids)
        self.assertAllEqual(output['attention_mask'], expected_attention_mask)
        self.assertAllEqual(output['token_type_ids'], expected_token_type_ids)

        print(self.bert_tokenizer(input_text))
