"""Model training code"""
"""
Usage Instructions:
  5-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=25 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1 --logdir=logs/omniglot5way/
  20-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=16 --k_shot=1 --n_way=20 --inner_update_lr=0.1 --num_inner_updates=5 --logdir=logs/omniglot20way/
  To run evaluation, use the '--meta_train=False' flag and the '--meta_test_set=True' flag to use the meta-test set.
"""

import csv
import numpy as np
import pickle
import random
import tensorflow as tf
from fillmore.maml import *
from fillmore.utils import *
from fillmore.dataset.load_text import TextDataGenerator

def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):

    with tf.GradientTape(watch_accessed_variables=False) as outer_tape:
        # only watch trainable variables
        outer_tape.watch(model.trainable_variables)

        result = model(inp, meta_batch_size=meta_batch_size,
                       num_inner_updates=num_inner_updates)

        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]
    
    gradients = outer_tape.gradient(
        total_losses_ts[-1], model.trainable_variables) # gradients return [NONE, NONE]
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(
        accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
    result = model(inp, meta_batch_size=meta_batch_size,
                   num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(
        accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def meta_train_fn(model, exp_string, data_generator, config,
                  n_way=5, meta_train_iterations=15000, meta_batch_size=25,
                  log=True, logdir='/tmp/data', k_shot=1, num_inner_updates=1, meta_lr=0.001):
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    pre_accuracies, post_accuracies = [], []

    num_classes = data_generator.num_classes

    optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

    writer = tf.summary.create_file_writer(logdir)
    with writer.as_default():
        for itr in range(meta_train_iterations):
            #############################
            #### YOUR CODE GOES HERE ####

            # sample a batch of training data and partition into
            # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
            # NOTE: The code assumes that the support and query sets have the same number of examples.

            #############################
            texts, labels = data_generator.sample_batch(
                config, "meta_train", meta_batch_size, shuffle=True, swap=False)
            B, N, K = texts.shape
            input_tr = tf.reshape(texts[:, :, :k_shot], [B, N*k_shot])
            input_ts = tf.reshape(texts[:, :, k_shot:], [B, N*k_shot])
            label_tr = tf.reshape(labels[:, :, :k_shot, :], [B, N*k_shot, -1])
            label_ts = tf.reshape(labels[:, :, k_shot:, :], [B, N*k_shot, -1])

            inp = (input_tr, input_ts, label_tr, label_ts)

            result = outer_train_step(
                inp, model, optimizer, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

            if itr % SUMMARY_INTERVAL == 0:
                pre_accuracies.append(result[-2])
                post_accuracies.append(result[-1][-1])
                tf.summary.scalar("pre-inner-loop-train-acc",
                                  np.mean(pre_accuracies), step=itr)
                tf.summary.scalar("post-inner-loop-test-acc",
                                  np.mean(post_accuracies), step=itr)
                writer.flush()

            if (itr != 0) and itr % PRINT_INTERVAL == 0:
                print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (
                    itr, np.mean(pre_accuracies), np.mean(post_accuracies))
                print(print_str)
                pre_accuracies, post_accuracies = [], []

            if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
                #############################
                #### YOUR CODE GOES HERE ####

                # sample a batch of validation data and partition it into
                # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
                # NOTE: The code assumes that the support and query sets have the same number of examples.

                #############################
                texts, labels = data_generator.sample_batch(
                    config, "meta_val", meta_batch_size, shuffle=True, swap=False)
                B, N, K = texts.shape
                input_tr = tf.reshape(
                    texts[:, :, :k_shot], [B, N*k_shot])
                input_ts = tf.reshape(
                    texts[:, :, k_shot:], [B, N*k_shot])
                label_tr = tf.reshape(
                    labels[:, :, :k_shot, :], [B, N*k_shot, -1])
                label_ts = tf.reshape(
                    labels[:, :, k_shot:, :], [B, N*k_shot, -1])

                inp = (input_tr, input_ts, label_tr, label_ts)
                result = outer_eval_step(
                    inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

                tf.summary.scalar(
                    "meta-val-pre-inner-loop-train-acc", result[-2], step=itr)
                tf.summary.scalar(
                    "meta-val-post-inner-loop-test-acc", result[-1][-1], step=itr)

                print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' %
                      (result[-2], result[-1][-1]))
                writer.flush()

    model_file = logdir + '/' + exp_string + '/model' + str(itr)
    print("Saving to ", model_file)
    model.save_weights(model_file)


# # calculated for omniglot
# NUM_META_TEST_POINTS = 600


def meta_test_fn(model, data_generator, config, n_way=5, meta_batch_size=25, k_shot=1,
                 num_inner_updates=1):

    NUM_META_TEST_POINTS = config.num_meta_test_points
    num_classes = data_generator.num_classes

    np.random.seed(1)
    random.seed(1)

    meta_test_accuracies = []

    for _ in range(NUM_META_TEST_POINTS):
        #############################
        #### YOUR CODE GOES HERE ####

        # sample a batch of test data and partition it into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same number of examples.

        #############################config, "meta_train", config.meta_batch_size
        texts, labels = data_generator.sample_batch(
            config, "meta_test", meta_batch_size, shuffle=True, swap=False)
        B, N, K = texts.shape
        input_tr = tf.reshape(texts[:, :, :k_shot], [B, N*k_shot])
        input_ts = tf.reshape(texts[:, :, k_shot:], [B, N*k_shot])
        label_tr = tf.reshape(labels[:, :, :k_shot], [B, N*k_shot, -1])
        label_ts = tf.reshape(labels[:, :, k_shot:], [B, N*k_shot, -1])

        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(
            inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        meta_test_accuracies.append(result[-1][-1])

    meta_test_accuracies = np.array(meta_test_accuracies)
    means = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)

    print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))


def run_maml(n_way=5, k_shot=1, meta_batch_size=25, meta_lr=0.001,
             inner_update_lr=0.4, num_filters=32, num_inner_updates=1,
             learn_inner_update_lr=False,
             resume=False, resume_itr=0, log=True, logdir='/tmp/data',
             data_path='./omniglot_resized', meta_train=True,
             meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1, num_meta_test_points=10, config=None):

    # model config
    if not config:
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.num_labels = n_way
        config.vocab_path = "/Users/yue.li/Desktop/repo/Personal/fillmore/tests/fixtures/vocab.txt"
        config.max_seq_len = 32
        config.mode="finetune" #TODO
        config.dataset = "reuters"
        config.n_train_class = 15
        config.n_val_class = 5
        config.n_test_class = 11
        config.data_path = "data/reuters.json"
        config.n_way = n_way
        config.k_shot = k_shot
        config.meta_batch_size = meta_batch_size
        config.num_meta_test_points = num_meta_test_points

    # call data_generator and get data with k_shot*2 samples per class
    data_generator = TextDataGenerator(config.n_way, config.k_shot*2, config.n_way, config.k_shot*2, config)
    texts, labels = data_generator.sample_batch(config, "meta_train", meta_batch_size, shuffle=True, swap=False)

    # set up MAML model
    model = MAML(config,
                 num_inner_updates=num_inner_updates,
                 inner_update_lr=inner_update_lr,
                 learn_inner_update_lr=learn_inner_update_lr)

    if meta_train_k_shot == -1:
        meta_train_k_shot = k_shot
    if meta_train_inner_update_lr == -1:
        meta_train_inner_update_lr = inner_update_lr

    exp_string = 'cls_'+str(n_way)+'.mbs_'+str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot) + '.inner_numstep_' + str(
        num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)

    # # save model
    # model_file = logdir + '/' + exp_string + '/model' + "0"
    # print("Saving to ", model_file)
    # model.save_weights(model_file)

    if meta_train:
        meta_train_fn(model, exp_string, data_generator, config,
                      n_way, meta_train_iterations, meta_batch_size, log, logdir,
                      k_shot, num_inner_updates, meta_lr)
    else:
        meta_batch_size = 1

        model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)

        meta_test_fn(model, data_generator, config, n_way,
                     meta_batch_size, k_shot, num_inner_updates)


if __name__ == "__main__":
    run_maml(n_way=5, k_shot=4,
             inner_update_lr=0.4,
             num_inner_updates=1,
             meta_train=True,
             meta_train_k_shot=1,
             learn_inner_update_lr=True,
             meta_train_iterations=10,
             meta_batch_size=2,
             logdir='./logs/maml',
             num_meta_test_points=10)
