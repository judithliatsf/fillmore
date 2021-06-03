import tensorflow as tf
from fillmore.metric_learners import MetricLearner, KNNLearner
from transformers import AutoConfig
from fillmore.protonet import ProtoLoss
from fillmore.bert_model import BertTextEncoderWrapper

class MetricLearnerTest(tf.test.TestCase):
    def setUp(self):
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.label_smoothing = 0.0
        config.max_seq_len = 32
        self.config = config
        embedding_func = BertTextEncoderWrapper(self.config)
        self.learner = MetricLearner(embedding_func)
        self.knn = KNNLearner(embedding_func)

    def test_forward_pass(self):
        episode = {
            "support_examples": [
                'am i allowed to put in a pto request for now to april',
                'can you set up a pto request for me for march 20th to april 12th',
                'can you put in a pto request for the 5th of june to june 8th',
                'i need a pto request for the dates jan 15th to jan 20th',
                'i need to put in a pto request from the 8th to the 17th of january'
            ],
            "query_examples": [
                'i need to put in a pto request from the 8th to the 17th of january',
                'can you put in a pto request for the 5th of june to june 8th'
            ],
            "support_labels_onehot": tf.one_hot([0, 0, 1, 2, 1], 3)
        }
        query_logits = self.learner(episode)
        self.assertAllEqual(query_logits.shape, [2, 3])
    
    def test_compute_logits(self):
        support_embeddings = tf.constant([[[1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], [[1.0, -1.0], [0.0, -1.0], [0.5, -0.5]]]) # [N*S, D]
        support_embeddings = tf.reshape(support_embeddings, [2*3, 2])
        query_embeddings = tf.constant([[[0.0, 0.5]], [[0.0, -0.5]]])
        query_embeddings = tf.reshape(query_embeddings, [2*1, 2])
        support_labels_onehot = tf.constant([[[1., 0.], [1., 0.], [1., 0.]], [[0., 1.], [0., 1.], [0., 1.]]])
        support_labels_onehot = tf.reshape(support_labels_onehot, [2*3, 2])
        query_labels_onehot = tf.constant([[[1., 0.]], [[1., 0.]]]) # [N, Q, N]
        query_labels_onehot = tf.reshape(query_labels_onehot, [2*1, 2])
        
        query_logits = self.learner.compute_logits_for_episode(support_embeddings, query_embeddings, support_labels_onehot)
        self.assertAllEqual(query_logits.shape, [2, 2])

        loss = self.learner.compute_loss(query_logits, query_labels_onehot)
        self.assertAllClose(loss, 1.006340)

        acc = self.learner.compute_accuracy(query_labels_onehot, query_logits)
        self.assertAllClose(acc, 0.5)

        query_prob = self.knn.compute_logits_for_episode(support_embeddings, query_embeddings, support_labels_onehot)
        knn_loss = self.knn.compute_loss(query_prob, query_labels_onehot)
        knn_acc = self.knn.compute_accuracy(query_labels_onehot, query_prob)

        self.assertAllClose(query_prob, [[1.,0.],[0.,1.]])
        self.assertAllClose(knn_loss, 7.689547)
        self.assertAllClose(knn_acc, 0.5)
    
    def test_proto_loss(self):
        x_latent = tf.constant([[[1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], [[1.0, -1.0], [0.0, -1.0], [0.5, -0.5]]]) # [N*S, D]
        x_latent = tf.reshape(x_latent, [2*3, 2])
        q_latent = tf.constant([[[0.0, 0.5]], [[0.0, -0.5]]])
        q_latent = tf.reshape(q_latent, [2*1, 2])
        num_classes = 2
        num_support = 3
        num_queries = 1
        query_labels_onehot = tf.constant([[[1., 0.]], [[1., 0.]]]) # [N, Q, N]
        support_labels_onehot = tf.constant([[[1., 0.], [1., 0.], [1., 0.]], [[0., 1.], [0., 1.], [0., 1.]]])
        support_labels_onehot = tf.reshape(support_labels_onehot, [2*3, 2])
        loss, acc = ProtoLoss(x_latent, q_latent, query_labels_onehot, num_classes, num_support, num_queries, self.config)
        print("loss : {}".format(loss)) # 1.0063408613204956
        print("acc: {}".format(acc)) # 0.5

import tensorflow as tf


class KNNLearner(tf.keras.Model):
    """A learner that uses a learned distance metric to make predictions."""

    def __init__(self, embedding_func, distance_type='l2', **kwargs):
        # overwrite base class attributes
        super(KNNLearner, self).__init__(**kwargs)
        self.embedding_func = embedding_func
        self.distance_type = distance_type
    
    def call(self, episode, query_examples=[]):
        if query_examples:
            episode["query_examples"] = query_examples
        query_prob = self.compute_logits_for_episode(episode)
        return query_prob
  
    def _compute_relevance(self, episode):
        """compute relevance/similarity between each query and support example
        """
        support_examples = episode['support_examples']
        query_examples = episode['query_examples']

        text_pairs = []
        for q_id in range(len(query_examples)):
            for s_id in range(len(support_examples)):
                text_pairs.append((query_examples[q_id], support_examples[s_id]))
        
        features = self.embedding_func.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs= text_pairs,
            padding = 'max_length',
            truncation= True,
            max_length= self.embedding_func.max_seq_len,
            return_token_type_ids=True,
            return_tensors= 'tf'
        )
        logits = self.embedding_func.compute_logits(features)
        relevance = logits[:, 0]
        relevance = tf.reshape(relevance, (len(query_examples), len(support_examples)))
        return relevance

    def compute_loss(self, query_prob, query_labels_onehot):
        """generate loss for an episode

        Args:
            query_prob ([N*Q, N]): query probability from `compute_logits_for_episode`
            query_labels_onehot ([N*Q, N]): onehot label for query
            support_labels_onehot ([N*S, N]): onehot label for support

        Returns:
            cost [float]: cross entropy loss averaged over all query examples
        """

#         bce = tf.keras.losses.BinaryCrossentropy()
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        cross_entropy_loss = bce(query_labels_onehot, query_prob)

        cost = tf.reduce_mean(cross_entropy_loss)
        return cost

    def compute_accuracy(self, query_labels_onehot, query_prob):
        """[summary]

        Args:
            query_labels_onehot ([N*Q, N]): onehot label for query
            query_prob ([N*Q, N]): query probability from `compute_logits_for_episode`

        Returns:
            accuracy [float]: accuracy averaged over all query examples
        """
        correct = tf.equal(tf.argmax(query_labels_onehot, -1), tf.argmax(query_prob, -1), -1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        return accuracy
      
    def compute_logits_for_episode(self, episode):
        """calculates the nearest neighbors prediction using the latent representation of
           support set and query set

        Args:
            support_embeddings ([N*S, D]): the latent representation of support set
            query_embeddings ([N*Q, D]): the latent representation of query set
            support_labels_onehot ([N*S, N]): the one-hot encodings of the query labels
        Returns:
            query_logits ([N*Q, N]): the query logits
        """
        if self.distance_type == 'l2':
            support_embeddings = self.embedding_func(episode["support_examples"])
            query_embeddings = self.embedding_func(episode["query_examples"])
            # [1, num_support, embed_dims]
            emb_support = tf.expand_dims(support_embeddings, axis=0)
            # [num_query, 1, embed_dims]
            emb_query = tf.expand_dims(query_embeddings, axis=1)
            # [num_query, num_support]
            distance = tf.norm(emb_support - emb_query, axis=2)
        elif self.distance_type == 'relevance':
            distance = -self._compute_relevance(episode)
        else:
            raise ValueError('Distance must be l2 or cosine')

        support_labels_onehot = episode["support_labels_onehot"]
        _, indices = tf.nn.top_k(-distance, k=1)
        indices = tf.squeeze(indices, axis=1)
        labels_x = tf.argmax(support_labels_onehot, axis=1)

        num_classes = support_labels_onehot.shape[1]
        labels_pred = tf.gather(tf.constant(labels_x), indices)
        labels_prob = tf.one_hot(labels_pred, depth=num_classes)
        return labels_prob