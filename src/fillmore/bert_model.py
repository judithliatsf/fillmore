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

    def call(self, inputs, weights=None, update_weights=True, training=False):
        # transform to features
        features = self.tokenizer(inputs)
        outputs = self.bert(features)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)

        if weights is None:
            # don't update weights
            logits = self.classifier(pooled_output)
        else:
            # update weights
            # set update=True will change the model weights
            if update_weights:
                # update weights for classifier
                self.classifier.set_weights(weights)
                logits = self.classifier(pooled_output)
            else:
                # create a new classifier layer
                classifier_copy = tf.keras.layers.Dense(
                    self.num_labels,
                    kernel_initializer=self.classifier.kernel_initializer,
                    name="classifier_copy")
                # initialize the weights
                classifier_copy.build((pooled_output.shape))
                classifier_copy.set_weights(weights)
                logits = classifier_copy(pooled_output)
        return logits
