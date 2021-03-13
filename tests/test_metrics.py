import tensorflow as tf
from fillmore.metrics import *
import numpy as np

class MetricsTest(tf.test.TestCase):
    def setUp(self):
        ## when threshold = 0.5, Cin=1, Nin=2, Coos=0, Noos=1, acc_in=1/2, oos_precision=0/1, oos_recall=0/1
        ## when threshold = 0.7, Cin=1, Nin=2, Coos=1, Noos=2, acc_in=1/2, oos_precision=1/2, oos_recall=1/1
        self.query_logits = tf.constant([[5.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
        self.query_labels_onehot = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.thresholds = [0.5, 0.7]
        self.oos_logits = tf.constant([[2.0, 3.0, 1.0]])
    
    def test_in_domain_stats_episode(self):
        in_domain_correct, oos_output = in_domain_stats_episode(self.query_logits, self.query_labels_onehot, self.thresholds)
        self.assertAllClose(in_domain_correct, np.array([[1., 0.],[0., 0.]]))
        self.assertAllClose(oos_output, np.array([[0., 0.],[0., 1.]]))
        self.assertEqual(in_domain_correct.shape, [2,2])
        self.assertEqual(oos_output.shape, [2,2])

    def test_oos_stats_episode(self):
        oos_correct = oos_stats_episode(self.oos_logits, self.thresholds)
        self.assertAllClose(oos_correct, np.array([[0., 1.]]))
        self.assertEqual(oos_correct.shape, [1,2])
    
    def test_oos_f1_batch(self):
        in_domain_correct, oos_output = in_domain_stats_episode(self.query_logits, self.query_labels_onehot, self.thresholds)
        oos_correct = oos_stats_episode(self.oos_logits, self.thresholds)
        in_domain_acc, oos_recall, oos_precision = oos_f1_batch(
        [in_domain_correct],
        [oos_correct],
        [oos_output])
        self.assertAllClose(in_domain_acc, np.array([0.5, 0.]))
        self.assertAllClose(oos_recall, np.array([0., 1.]))
        self.assertAllClose(oos_precision, np.array([0. , 0.5]))
        self.assertEqual(in_domain_acc.shape, (2,))
        self.assertEqual(oos_recall.shape, (2,))
        self.assertEqual(oos_precision.shape, (2,))
        