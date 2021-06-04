from numpy.core.numeric import cross
from transformers import TFPreTrainedModel, TFRobertaModel
from transformers import RobertaTokenizer
from re import T
from transformers import TFBertPreTrainedModel, TFBertMainLayer
from transformers.models.bert.tokenization_bert import BertTokenizer
from fillmore.bert_featurizer import BertSingleSentenceFeaturizer
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFPreTrainedModel, AutoConfig
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig


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
        self.max_seq_len = config.max_seq_len
        self.encoder = BertTextEncoder(config)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dummy_text_inputs = [
            "cats are in the cloud",
            "dogs are on the ground"
        ]

    def call(self, inputs):
        features = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=inputs,
            truncation=True,
            padding=True, max_length=self.max_seq_len, return_tensors='tf')
        outputs = self.encoder(features)
        return outputs


class RobertaEncoder(TFPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.max_seq_len = config.max_seq_len
        self.encoder = TFRobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.dummy_text_inputs = [
            "cats are in the cloud",
            "dogs are on the ground"
        ]

    def call(self, inputs):
        features = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=inputs,
            truncation=True,
            padding=True, max_length=self.max_seq_len, return_tensors='tf')
        outputs = self.encoder(features)[1]
        return outputs


class RobertaBinaryClassifier(TFPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.max_seq_len = config.max_seq_len
        self.encoder = TFRobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.num_classes = 2
        self.binary_classifier = tf.keras.layers.Dense(
            self.num_classes,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=config.initializer_range),
            name="classifier"
        )
        self.dummy_text_inputs = self.tokenizer.encode_plus(
        text="cats are in the cloud", 
        text_pair="dogs are on the ground",
        truncation=True,
        padding='max_length', max_length=config.max_seq_len,
        return_tensors='tf', return_token_type_ids=True)
        self.dummy_labels = [0, 1] # entail or not entail

    def call(self, features, training=False):
        # inputs may be of shape [batch_size, 1, max_seq_len]
        input_ids = tf.squeeze(
            features['input_ids'], axis=1) if len(features['input_ids'].shape) == 3 else features['input_ids']
        token_type_ids = tf.squeeze(
            features['token_type_ids'], axis=1) if len(features['token_type_ids'].shape) == 3 else features['token_type_ids']
        attention_mask = tf.squeeze(
            features['attention_mask'], axis=1) if len(features['attention_mask'].shape) == 3 else features['attention_mask']
        pooled_output = self.encoder(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[1]

        return pooled_output

    def compute_logits(self, inputs, training=False):
        pooled_output = self.call(inputs)
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.binary_classifier(pooled_output)
        return logits

    def compute_loss(self, inputs, labels_one_hot):
        if len(labels_one_hot.shape) != 2:
            labels_one_hot = tf.one_hot(labels_one_hot, depth=self.num_classes)
        logits = self.compute_logits(inputs, training=True)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(labels_one_hot)))

    def compute_acc(self, inputs, labels_one_hot):
        if len(labels_one_hot.shape) != 2:
            labels_one_hot = tf.one_hot(labels_one_hot, depth=self.num_classes)
        logits = self.compute_logits(inputs, training=False)
        prob = tf.nn.softmax(logits)
        predictions = tf.argmax(prob, -1)  # (B, 1)
        labels = tf.argmax(labels_one_hot, -1)  # (B, 1)
        return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))

class AutoBinaryClassifier(TFPreTrainedModel):
    def __init__(self, config, model_name, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.max_seq_len = config.max_seq_len
        self.encoder = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        self.num_classes = self.encoder.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dummy_text_inputs = self.tokenizer.encode_plus(
        text="cats are in the cloud", 
        text_pair="dogs are on the ground",
        truncation=True,
        padding='max_length', max_length=config.max_seq_len,
        return_tensors='tf', return_token_type_ids=True)
        self.max_seq_len = config.max_seq_len
        self.dummy_labels = [0, 1] # entail or not entail

    def call(self, features, training=False):
        # inputs may be of shape [batch_size, 1, max_seq_len]
        input_ids = tf.squeeze(
            features['input_ids'], axis=1) if len(features['input_ids'].shape) == 3 else features['input_ids']
        token_type_ids = tf.squeeze(
            features['token_type_ids'], axis=1) if len(features['token_type_ids'].shape) == 3 else features['token_type_ids']
        attention_mask = tf.squeeze(
            features['attention_mask'], axis=1) if len(features['attention_mask'].shape) == 3 else features['attention_mask']
        logits = self.encoder(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, training=training).logits

        return logits

    def compute_logits(self, inputs, training=False):
        
        return self.call(inputs, training=training)

    def compute_loss(self, inputs, labels_one_hot):
        if len(labels_one_hot.shape) != 2:
            labels_one_hot = tf.one_hot(labels_one_hot, depth=self.num_classes)
        logits = self.compute_logits(inputs, training=True)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(labels_one_hot)))

    def compute_acc(self, inputs, labels_one_hot):
        if len(labels_one_hot.shape) != 2:
            labels_one_hot = tf.one_hot(labels_one_hot, depth=self.num_classes)
        logits = self.compute_logits(inputs, training=False)
        prob = tf.nn.softmax(logits)
        predictions = tf.argmax(prob, -1)  # (B, 1)
        labels = tf.argmax(labels_one_hot, -1)  # (B, 1)
        return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))
    
    def text_to_feature(self, inputs=[("cats are in the cloud", "dogs are on the ground")]):
        """Give a list of tuple of texts, output features"""
        return self.tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=inputs,
        truncation=True,
        padding='max_length', max_length=self.max_seq_len,
        return_tensors='tf', return_token_type_ids=True)

if __name__ == "__main__":
    pretrained_model_name_or_path = "roberta"
    config = RobertaConfig()
    config.max_seq_len = 16
    model = RobertaEncoder(config)
    output = model(model.dummy_text_inputs)
    config = RobertaConfig(max_seq_len=16)
    cross_encoder = RobertaBinaryClassifier(config)
    features = cross_encoder(cross_encoder.dummy_text_inputs)
    cross_encoder = AutoBinaryClassifier(config, model_name="bert-base-cased-finetuned-mrpc")
    logits = cross_encoder(cross_encoder.dummy_text_inputs)
