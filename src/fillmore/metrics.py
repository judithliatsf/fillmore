import tensorflow as tf
import numpy as np

def oos_stats_episode(oos_logits, thresholds):
    """Intermediate statistics used to compute oos_f1 given logits output for oos episode

    Args:
        oos_logits (tf.Tensor of [Q, N]): logits output for oos_examples
        thresholds (List[float]): possible candidates as threshold between (0,1)

    Returns:
        oos_correct (tf.Tensor of [Q, T]: oos_correct[i][j] = 1.0 if the ith example
                                        is correctly identified as oos under the 
                                        thresholds[j]
    """
    oos_prob = tf.nn.softmax(oos_logits)
    oos_conf = tf.expand_dims(tf.reduce_max(oos_prob, -1), axis=1)
    oos_correct = tf.less(oos_conf, thresholds)
    oos_correct = tf.cast(oos_correct, tf.float32)
    return oos_correct

def in_domain_stats_episode(query_logits, query_labels_onehot, thresholds):
    """Intermediate statistics for query episode used to compute oos_f1

    Args:
        query_logits (tf.Tensor of [Q, N]): output logits of query episode
        query_labels_onehot (tf.Tensor of [Q, N]): one hot ground truth label
        thresholds (List[float]): possible candidates as threshold between (0,1)

    Returns:
        in_domain_correct (tf.Tensor of [Q, T]): in_domain_correct[i][j] = 1.0 if 
                                        the ith example is correctly classified 
                                        and its confidence >= thresholds[j]
        oos_output (tf.Tensor of [Q, T]): oos_output[i][j] = 1.0 if 
                                        the ith example is classified as OOS as 
                                        its confidence < thresholds[j]
    """
    query_prob = tf.nn.softmax(query_logits)
    query_pred = tf.expand_dims(tf.cast(tf.argmax(query_prob, -1), tf.float32),axis=1) #[Q, 1]
    query_conf = tf.expand_dims(tf.reduce_max(query_prob, -1), axis=1) # [Q, 1]
    query_conf_mask = tf.cast(tf.greater_equal(query_conf, thresholds), tf.float32) # [Q, T]
    query_pred = tf.multiply(query_pred, query_conf_mask) #[Q, T]
    query_label = tf.cast(tf.argmax(query_labels_onehot, -1), tf.float32) # [Q,]
    in_domain_correct = tf.equal(query_label, query_pred)
    in_domain_correct = tf.cast(in_domain_correct, tf.float32)
    oos_output = tf.cast(tf.less(query_conf, thresholds), tf.float32)
    # accuracy = tf.reduce_mean(in_domain_correct)
    return in_domain_correct, oos_output

def oos_f1_batch(in_domain_correct_all, oos_correct_all, oos_output_all):
    """For a batch of in_domain_correct, oos_correct and oos_output, compute oos_f1

    Args:
        in_domain_correct_all (List[in_domain_correct]): [B, Qin, T]
        oos_correct_all (List[oos_correct]): [B, Qoos, T]
        oos_output_all (List[oos_output]): [B, Qin, T]

    Returns:
        in_domain_acc [type]: [description]
    """
    in_domain_correct_all = np.vstack(in_domain_correct_all)
    oos_correct_all = np.vstack(oos_correct_all)
    
    in_domain_acc = np.mean(in_domain_correct_all, axis=0)
    oos_recall = np.mean(oos_correct_all, axis=0)

    oos_output_all = np.vstack(oos_output_all)
    oos_total_all = tf.concat([oos_output_all, oos_correct_all], axis=0)
    oos_correct = np.sum(oos_correct_all, axis=0, keepdims=True) # [1, T]
    oos_total = np.sum(oos_total_all, axis=0, keepdims=True) + np.finfo(np.float32).eps # [1,T]
    oos_precision = np.mean(np.multiply(oos_correct, 1.0/oos_total), axis=0)
    return in_domain_acc, oos_recall, oos_precision

if __name__ == "__main__":
    query_logits = tf.constant([[5.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    query_labels_onehot = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    thresholds = [0.5, 0.7]
    in_domain_correct, oos_output = in_domain_stats_episode(query_logits, query_labels_onehot, thresholds)
    print(in_domain_correct) # [[1., 0.],[0., 0.]]
    print(oos_output) # [[0., 0.],[0., 1.]]
    oos_logits = tf.constant([[2.0, 3.0, 1.0]])
    oos_correct = oos_stats_episode(oos_logits, thresholds)
    print(oos_correct) # [[0., 0.],[0., 1.]]
    in_domain_acc, oos_recall, oos_precision = oos_f1_batch(
        [in_domain_correct],
        [oos_correct],
        [oos_output])
    ## when threshold = 0.5, Cin=1, Nin=2, Coos=0, Noos=1, acc_in=1/2, oos_precision=0/1, oos_recall=0/1
    ## when threshold = 0.7, Cin=1, Nin=2, Coos=1, Noos=2, acc_in=1/2, oos_precision=1/2, oos_recall=1/1
    import pdb; pdb.set_trace()