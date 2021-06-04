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
        self.episode = {
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
            "support_labels_onehot": tf.one_hot([0, 0, 1, 2, 1], 3),
            "query_labels_onehot": tf.one_hot([0, 0], 3)
        }
    def test_forward_pass(self):
        query_logits = self.learner(self.episode)
        self.assertAllEqual(query_logits.shape, [2, 3])
    
    def test_proto(self):
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
    
    def test_knn(self):
        query_prob = self.knn.compute_logits_for_episode(self.episode)
        knn_loss = self.knn.compute_loss(query_prob, self.episode["query_labels_onehot"])
        knn_acc = self.knn.compute_accuracy(self.episode["query_labels_onehot"], query_prob)
        self.assertAllEqual(query_prob.shape, [2, 3])
        self.assertAllClose(knn_loss.numpy(), 10.252728)
        self.assertAllClose(knn_acc.numpy(), 0.0)
    
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
