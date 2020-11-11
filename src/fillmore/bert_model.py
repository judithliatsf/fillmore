from transformers import TFBertPreTrainedModel, TFBertMainLayer
from fillmore.bert_featurizer import BertSingleSentenceFeaturizer
import tensorflow as tf
from transformers import AutoTokenizer


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
        self.max_seq_len = config.max_seq_len
        self.dummy_text_inputs = {
        'input_ids': tf.constant([[101, 2365, 1997, 13259, 102], [101, 2365, 1997, 103, 102]], dtype=tf.int32), 
        'token_type_ids': tf.constant([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=tf.int32), 
        'attention_mask': tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=tf.int32)}

        # layer definition
        self.bert = TFBertMainLayer(config, name="bert")

    def call(self, inputs):
        outputs = self.bert(inputs)
        pooled_output = outputs[1]
        return pooled_output

class BertTextEncoderWrapper(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.max_seq_len = config.max_seq_len
        self.encoder = BertTextEncoder(config)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        features = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=inputs, 
                    truncation=True, 
                    padding=True, max_length=self.max_seq_len, return_tensors='tf')
        outputs = self.encoder(features)
        return outputs