from transformers import TFBertPreTrainedModel, TFBertMainLayer
from fillmore.bert_featurizer import BertSingleSentenceFeaturizer
import tensorflow as tf


class BertTextClassification(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.vocab_path = config.vocab_path
        self.max_seq_len = config.max_seq_len
        self.tokenizer = BertSingleSentenceFeaturizer(
            self.vocab_path,
            max_sen_len=self.max_seq_len
        )
        self.dummy_text_inputs = tf.constant(
            ["the son of flynn", "more than just a man"])

        # layer definition
        self.bert = TFBertMainLayer(config, name="bert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=config.initializer_range),
            name="classifier"
        )

    def call(self, inputs, training=False):
        # transform to features
        features = self.tokenizer(inputs)
        outputs = self.bert(features)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits

class BertTextEncoder(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.vocab_path = config.vocab_path
        self.max_seq_len = config.max_seq_len
        self.tokenizer = BertSingleSentenceFeaturizer(
            self.vocab_path,
            max_sen_len=self.max_seq_len
        )
        self.dummy_text_inputs = tf.constant(
            ["the son of flynn", "more than just a man"])

        # layer definition
        self.bert = TFBertMainLayer(config, name="bert")

    def call(self, inputs, training=False):
        # transform to features
        features = self.tokenizer(inputs)
        outputs = self.bert(features)
        pooled_output = outputs[1]
        return pooled_output
