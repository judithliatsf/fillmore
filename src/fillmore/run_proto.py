# run_ProtoNet
from logging import config
import numpy as np
import tensorflow as tf
import os
from fillmore.protonet import ProtoLoss
from fillmore.bert_model import BertTextEncoder
from fillmore.dataset.load_text import TextDataGenerator
from fillmore.nn import nn_eval
from transformers import AutoConfig
from transformers import AutoTokenizer
import copy

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def proto_net_train_step(model, optim, x, q, labels_ph, config):
    num_classes, num_support = x.shape
    num_queries = q.shape[1]
    x = x.reshape([num_classes*num_support, ]).tolist()
    q = q.reshape([num_classes*num_queries, ]).tolist()

    x = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=x,
        truncation=True, padding=True, max_length=model.max_seq_len, return_tensors='tf')
    q = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=q,
        truncation=True, padding=True, max_length=model.max_seq_len, return_tensors='tf')
    labels_ph = tf.constant(labels_ph)

    with tf.GradientTape() as tape:
        x_latent = model(x)  # [N*S, D]
        q_latent = model(q)  # [N*Q, D]
        ce_loss, acc = ProtoLoss(
            x_latent, q_latent, labels_ph, num_classes, num_support, num_queries, config)

    gradients = tape.gradient(ce_loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    return ce_loss, acc


def proto_net_eval(model, x, q, labels_ph, config):
    num_classes, num_support = x.shape
    num_queries = q.shape[1]
    x = x.reshape([num_classes*num_support, ]).tolist()
    q = q.reshape([num_classes*num_queries, ]).tolist()

    x = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=x,
        truncation=True, padding=True, max_length=model.max_seq_len, return_tensors='tf')
    q = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=q,
        truncation=True, padding=True, max_length=model.max_seq_len, return_tensors='tf')
    labels_ph = tf.constant(labels_ph)

    x_latent = model(x)
    q_latent = model(q)
    ce_loss, acc = ProtoLoss(
        x_latent, q_latent, labels_ph, num_classes, num_support, num_queries, config)

    return ce_loss, acc


def run_protonet(config, n_way=20, k_shot=1, n_query=5,
                 n_meta_test_way=20, k_meta_test_shot=5, n_meta_test_query=5,
                 logdir="./proto", meta_train=True,
                 n_epochs=20, n_episodes=100, n_meta_test_episodes=1000,
                 checkpoint_name=None, lr_scheduler=False,
                 smlmt=False, smlmt_ratio=0.5, smlmt_data_path="data/smlmt_clinc150small.json"):

    exp_string = 'cls_'+str(n_way)+'.eps_'+str(n_episodes) + \
        '.k_shot_' + str(k_shot) + '.n_query_' + str(n_query)
    model = BertTextEncoder(config)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        clipnorm=config.clipnorm
    )

    # call DataGenerator with k_shot+n_query samples per class
    data_generator = TextDataGenerator(
        n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query, config)

    if smlmt:
        smlmt_config = copy.deepcopy(config)
        smlmt_config.dataset = "smlmt"
        smlmt_config.data_path = smlmt_data_path
        smlmt_config.oos = False
        smlmt_generator = TextDataGenerator(
            n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query, smlmt_config)

    if meta_train:

        config.save_pretrained(os.path.join(logdir, exp_string))

        writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
        with writer.as_default():
            itr = 0
            for ep in range(n_epochs):
                for epi in range(n_episodes):
                    # set learning rate
                    if (epi+1) % 50 == 0:
                        if lr_scheduler == True:
                            optimizer.learning_rate = optimizer.learning_rate/2
                            print("learning rate for itration {} : {}".format(
                                itr, optimizer.learning_rate))
                    #############################
                    # sample a batch of training data and partition it into
                    # support and query sets
                    if not smlmt:
                        texts_train, labels_train = data_generator.sample_batch(
                            config, 'meta_train', 1)
                    else:
                        lucky_number = np.random.random_sample()
                        if lucky_number < smlmt_ratio:
                            texts_train, labels_train = smlmt_generator.sample_batch(
                                smlmt_config, 'meta_train', 1)
                        else:
                            texts_train, labels_train = data_generator.sample_batch(
                                config, 'meta_train', 1)
                    texts = texts_train[0]
                    labels = labels_train[0]
                    N, K = texts.shape
                    support = texts[:, :k_shot].reshape((N, k_shot))
                    query = texts[:, k_shot:].reshape((N, n_query))
                    label_s = labels[:, :k_shot, :].reshape((N, k_shot, -1))
                    label_q = labels[:, k_shot:, :].reshape((N, n_query, -1))
                    #############################
                    ls, ac = proto_net_train_step(
                        model, optimizer, x=support, q=query, labels_ph=label_q)
                    tf.summary.scalar("meta-train-loss", ls.numpy(), step=itr)
                    tf.summary.scalar("meta-train-acc", ac.numpy(), step=itr)

                    ############################# Meta Validation #############################
                    if (epi+1) % 10 == 0:
                        #############################
                        # sample a batch of validation data and partition it into
                        # support and query sets
                        texts_val, labels_val = data_generator.sample_batch(
                            config, 'meta_val', 1)
                        texts = texts_val[0]
                        labels = labels_val[0]
                        N, K = texts.shape
                        support = texts[:, :k_shot].reshape((N, k_shot))
                        query = texts[:, k_shot:].reshape((N, n_query))
                        label_s = labels[:, :k_shot, :].reshape(
                            (N, k_shot, -1))
                        label_q = labels[:, k_shot:, :].reshape(
                            (N, n_query, -1))
                        ############################# OOS Validation #############################
                        if config.oos == True:
                            oos_texts_val, oos_labels_val = data_generator.sample_batch(
                              config, 'oos_val', 1
                            )
                            oos_query = oos_texts_val[0][:, k_shot:].reshape((1, n_query))
                            oos_label_q = oos_labels_val[0][:,k_shot:, :].reshape((1, n_query, -1))
                            print(oos_query.shape)
                        #############################
                        if config.nn_eval:
                            pred, ls, ac = nn_eval(
                                model, tokenizer, x=support, q=query, labels_x=label_s, labels_q=label_q)
                        else:
                            val_ls, val_ac = proto_net_eval(
                                model, x=support, q=query, labels_ph=label_q, config=config)
                            oos_val_loss, oos_val_acc = proto_net_eval(
                              model, x=support, q=oos_query, labels_ph=oos_label_q, config=config)
                        tf.summary.scalar(
                            "meta-val-loss", val_ls.numpy(), step=itr)
                        tf.summary.scalar(
                            "meta-val-acc", val_ac.numpy(), step=itr)
                        print('[epoch {}/{}, episode {}/{}] => meta-training loss: {:.5f}, meta-training acc: {:.5f}, meta-val loss: {:.5f}, meta-val acc: {:.5f}'.format(ep+1,
                                                                                                                                                                          n_epochs,
                                                                                                                                                                          epi+1,
                                                                                                                                                                          n_episodes,
                                                                                                                                                                          ls,
                                                                                                                                                                          ac,
                                                                                                                                                                          val_ls,
                                                                                                                                                                          val_ac))
                        writer.flush()
                    itr = itr + 1
                    ############ end of episode ################
                # save model at the end of every epoch
                model_file = logdir + "/" + exp_string + '/model' + str(ep)
                print("Saving to ", model_file)
                model.save_weights(model_file)
                ################# end of epochs ######################

    ############################# Meta Testing #############################
    print('Testing...')
    checkpoint_dir = logdir + '/' + exp_string
    if checkpoint_name is not None:
        model_file = os.path.join(checkpoint_dir, checkpoint_name)
    else:
        model_file = tf.train.latest_checkpoint(checkpoint_dir)
    print("Restoring model weights from ", model_file)
    model.load_weights(model_file)

    meta_test_accuracies = []
    for epi in range(n_meta_test_episodes):
        #############################
        # sample a batch of test data and partition it into
        # support and query sets
        texts_test, labels_test = data_generator.sample_batch(
            config, 'meta_test', 1)
        texts = texts_test[0]
        labels = labels_test[0]
        N, K = texts.shape
        support = texts[:, :k_meta_test_shot].reshape(
            [n_meta_test_way, k_meta_test_shot])
        query = texts[:, k_meta_test_shot:].reshape(
            [n_meta_test_way, n_meta_test_query])
        label_s = labels[:, :k_meta_test_shot, :].reshape(
            [n_meta_test_way, k_meta_test_shot, -1])
        label_q = labels[:, k_meta_test_shot:, :].reshape(
            [n_meta_test_way, n_meta_test_query, -1])
        
        if config.oos == True:
            oos_texts_val, oos_labels_val = data_generator.sample_batch(
              config, 'oos_test', 1
            )
            # same size as query and label_q
            oos_query = oos_texts_val[0][:, k_meta_test_shot:].reshape((1, n_meta_test_query))
            oos_label_q = oos_labels_val[0][:,k_meta_test_shot:, :].reshape((1, n_meta_test_query, -1))
            print(oos_query.shape)
        #############################
        ls, ac = proto_net_eval(model, x=support, q=query, labels_ph=label_q, config=config)
        meta_test_accuracies.append(ac)
        if (epi+1) % 5 == 0:
            print('[meta-test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi +
                                                                                  1, n_meta_test_episodes, ls, ac))
    avg_acc = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    print(
        'Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.mode = "proto"
    config.dataset = "clinc150a"
    config.data_path = "data/clinc150small.json"
    config.vocab_path = "data/vocab.txt"

    config.max_seq_len = 16
    config.learning_rate = 1e-5
    config.epsilon = 1e-08
    config.clipnorm = 1.0
    config.num_examples_from_class = 20
    config.label_smoothing = 0.2

    config.n_way = 5
    config.k_shot = 10
    config.n_query = 10
    config.n_meta_test_way = 5
    config.k_meta_test_shot = 10
    config.n_meta_test_query = 10
    config.n_episodes = 4
    config.n_epochs = 3
    config.n_meta_test_episodes = 2
    config.lr_scheduler = True
    config.nn_eval = True

    config.oos = True
    config.oos_data_path = "data/clinc150_oos.json"
    logdir = "./logs/proto"

    run_protonet(config, n_way=config.n_way, k_shot=config.k_shot, n_query=config.n_query,
                 n_meta_test_way=config.n_meta_test_way, k_meta_test_shot=config.k_meta_test_shot, n_meta_test_query=config.n_meta_test_query,
                 logdir=logdir, meta_train=False,
                 n_epochs=config.n_epochs, n_episodes=config.n_episodes, n_meta_test_episodes=config.n_meta_test_episodes,
                 checkpoint_name=None, lr_scheduler=config.lr_scheduler,
                 smlmt=True, smlmt_ratio=0.5, smlmt_data_path="data/smlmt_clinc150small.json")
