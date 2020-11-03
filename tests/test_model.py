import tensorflow as tf
from fillmore.bert_featurizer import BertSingleSentenceFeaturizer
from fillmore.bert_model import BertTextClassification
from fillmore.utils import cross_entropy_loss
from transformers import *
import os

class BertSingleSentenceFeaturizerTest(tf.test.TestCase):
    def setUp(self):
        self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/vocab.txt")
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased")
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        config = self.model.config

        # add featurizer configuration
        config.num_labels = 3
        config.vocab_path = self.vocab_path
        config.max_seq_len = 7
        self.config = config

    def test_bert_featurizer(self):
        tokenizer = BertSingleSentenceFeaturizer(
            self.vocab_path,
            max_sen_len=self.config.max_seq_len
        )
        
        input_text = [
            'mary hopped.',
            'jane skipped.',
            'who farted?',
            'the quick brown fox jumped over the lazy dogs.'
        ]
        input_tensor = tf.constant(input_text)

        output = tokenizer(input_tensor)
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
        self.assertAllEqual(output['input_ids'], expected_token_ids)
        self.assertAllEqual(output['attention_mask'], expected_attention_mask)
        self.assertAllEqual(output['token_type_ids'], expected_token_type_ids)

    def test_bert_featurizer_and_model(self):
        # inputs for test
        input_text = [
            'mary hopped.',
            'jane skipped.',
            'who farted?',
            'the quick brown fox jumped over the lazy dogs.'
        ]
        input_tensor = tf.constant(input_text)
        input_labels = tf.one_hot([0,1,2,1], depth=self.config.num_labels)

        # test inference
        tf.random.set_seed(12345)
        wrapper = BertTextClassification(self.config)

        # forward run to get logits and loss
        logits = wrapper(input_tensor)
        loss = cross_entropy_loss(logits, input_labels)


        expected_logits = tf.constant([[ 0.45390517, -0.04176356, 0.25615168],
                                       [ 0.4850534,  -0.03088464,  0.24046643],
                                       [ 0.45687744, -0.03055511,  0.25601646],
                                       [ 0.4522503,  -0.09165739, 0.25520837]], dtype=tf.float32)
        
        self.assertAllClose(logits, expected_logits)
        self.assertAllEqual(logits.shape, expected_logits.shape)
        self.assertAllClose(loss, 1.195133090019226)

    def test_update_weights(self):
        wrapper = BertTextClassification(self.config)
        # initialize weights
        old_logits = wrapper(wrapper.dummy_text_inputs)

        # check weights
        self.assertEqual(len(wrapper.weights), 201)
        self.assertEqual(len(wrapper.classifier.weights), 2)

        # change weights and test inference
        old_weights = wrapper.classifier.get_weights()
        new_weights = []
        for w in old_weights:
            new_weights.append(tf.zeros_like(w))
        
        expected_logits = tf.constant(
            [[0., 0., 0.],
             [0., 0., 0.]], dtype=tf.float32
        )

        # run inference without change model weights
        logits = wrapper(wrapper.dummy_text_inputs, new_weights, update_weights=False)
        self.assertAllClose(logits, expected_logits)
        self.assertAllClose(wrapper.classifier.get_weights(), old_weights)
        logits1 = wrapper(wrapper.dummy_text_inputs)
        self.assertAllClose(logits1, old_logits)

        # run inference by changing the existing model weights
        logits = wrapper(wrapper.dummy_text_inputs, new_weights, update_weights=True)
        self.assertAllClose(logits, expected_logits)
        self.assertAllClose(wrapper.classifier.get_weights(), new_weights)

