from transformers import TFBertPreTrainedModel, TFBertForSequenceClassification
from fillmore.bert_featurizer import BertSingleSentenceFeaturizer

class BertTextClassification(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.vocab_path = config.vocab_path
        self.max_seq_len = config.max_seq_len
        self.model = TFBertForSequenceClassification(config)
        self.tokenizer = BertSingleSentenceFeaturizer(
            self.vocab_path,
            max_sen_len=self.max_seq_len
        )
        self.updated_layer = self.model.classifier
        self.updated_weights = self.updated_layer.weights
    
    def call(self, inputs, weights=None, labels=None):
        # update weights
        if weights is not None:
            # update weights for classifier
            self.updated_layer.set_weights(weights)

        # inference
        features = self.tokenizer(inputs)
        return self.model(features, labels=labels)