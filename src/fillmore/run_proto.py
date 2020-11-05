# run_ProtoNet
from logging import config
import numpy as np
import tensorflow as tf
import os
from fillmore.protonet import ProtoLoss
from fillmore.bert_model import BertTextEncoder
from fillmore.dataset.load_text import TextDataGenerator
from transformers import AutoConfig

def proto_net_train_step(model, optim, x, q, labels_ph):
  num_classes, num_support = x.shape
  num_queries = q.shape[1]
  x = tf.reshape(x, [num_classes*num_support,])
  q = tf.reshape(q, [num_classes*num_queries,])

  with tf.GradientTape() as tape:
    x_latent = model(x) # [N*S, D]
    q_latent = model(q) # [N*Q, D]
    ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

  gradients = tape.gradient(ce_loss, model.trainable_variables)
  optim.apply_gradients(zip(gradients, model.trainable_variables))
  return ce_loss, acc

def proto_net_eval(model, x, q, labels_ph):
  num_classes, num_support = x.shape
  num_queries = q.shape[1]
  x = tf.reshape(x, [num_classes*num_support,])
  q = tf.reshape(q, [num_classes*num_queries,])

  x_latent = model(x)
  q_latent = model(q)
  ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

  return ce_loss, acc 

def run_protonet(config, n_way=20, k_shot=1, 
                 n_query=5, n_meta_test_way=20, k_meta_test_shot=5, 
                 n_meta_test_query=5, logdir="./proto", meta_train=True,
                 n_epochs=20, n_episodes=100, n_meta_test_episodes = 1000):

  exp_string = 'cls_'+str(n_way)+'.eps_'+str(n_episodes) + '.k_shot_' + str(k_shot) + '.n_query_' + str(n_query)
  model = BertTextEncoder(config)
  optimizer = tf.keras.optimizers.Adam()

    # call DataGenerator with k_shot+n_query samples per class
  data_generator = TextDataGenerator(n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query, config)
  
  if meta_train:
    writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
    with writer.as_default():
      itr = 0
      for ep in range(n_epochs):
        for epi in range(n_episodes):
          #############################
          #### YOUR CODE GOES HERE ####

          # sample a batch of training data and partition it into
          # support and query sets
          texts_train, labels_train = data_generator.sample_batch(config, 'meta_train', 1)
          texts = texts_train[0]
          labels = labels_train[0]
          N, K = texts.shape
          support = tf.reshape(texts[:, :k_shot], [N, k_shot])
          query = tf.reshape(texts[:, k_shot:], [N, n_query])
          label_s = tf.reshape(labels[:, :k_shot, :], [N, k_shot, -1])
          label_q = tf.reshape(labels[:, k_shot:, :], [N, n_query, -1])
          #############################
          ls, ac = proto_net_train_step(model, optimizer, x=support, q=query, labels_ph=label_q)
          tf.summary.scalar("meta-train-loss", ls.numpy(), step=itr)
          tf.summary.scalar("meta-train-acc", ac.numpy(), step=itr)

          if (epi+1) % 50 == 0:
            #############################
            #### YOUR CODE GOES HERE ####

            # sample a batch of validation data and partition it into
            # support and query sets
            texts_val, labels_val = data_generator.sample_batch(config, 'meta_val', 1)
            texts = texts_val[0]
            labels = labels_val[0]
            N, K = texts.shape
            support = tf.reshape(texts[:, :k_shot], [N, k_shot])
            query = tf.reshape(texts[:, k_shot:], [N, n_query])
            label_s = tf.reshape(labels[:, :k_shot, :], [N, k_shot, -1])
            label_q = tf.reshape(labels[:, k_shot:, :], [N, n_query, -1])
            #############################
            val_ls, val_ac = proto_net_eval(model, x=support, q=query, labels_ph=label_q)
            tf.summary.scalar("meta-val-loss", val_ls.numpy(), step=itr)
            tf.summary.scalar("meta-val-acc", val_ac.numpy(), step=itr)
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
        # save model at the end of every epoch        
        model_file = logdir + "/" + exp_string + '/model' + str(ep)
        print("Saving to ", model_file)
        model.save_weights(model_file)

  print('Testing...')
  model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
  print("Restoring model weights from ", model_file)
  model.load_weights(model_file)
  
  meta_test_accuracies = []
  for epi in range(n_meta_test_episodes):
    #############################
    #### YOUR CODE GOES HERE ####

    # sample a batch of test data and partition it into
    # support and query sets
    texts_test, labels_test = data_generator.sample_batch(config, 'meta_test', 1)
    texts = texts_test[0]
    labels = labels_test[0]
    N, K = texts.shape
    support = tf.reshape(texts[:, :k_meta_test_shot], [n_meta_test_way, k_meta_test_shot])
    query = tf.reshape(texts[:, k_meta_test_shot:], [n_meta_test_way, n_meta_test_query])
    label_s = tf.reshape(labels[:, :k_meta_test_shot, :], [n_meta_test_way, k_meta_test_shot, -1])
    label_q = tf.reshape(labels[:, k_meta_test_shot:, :], [n_meta_test_way, n_meta_test_query, -1])
    #############################
    ls, ac = proto_net_eval(model, x=support, q=query, labels_ph=label_q)
    meta_test_accuracies.append(ac)
    if (epi+1) % 50 == 0:
      print('[meta-test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_meta_test_episodes, ls, ac))
  avg_acc = np.mean(meta_test_accuracies)
  stds = np.std(meta_test_accuracies)
  print('Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))

if __name__ == "__main__":
  config = AutoConfig.from_pretrained("bert-base-uncased")
  config.mode="proto"
  config.dataset = "reuters"
  config.n_train_class = 15
  config.n_val_class = 5
  config.n_test_class = 11
  config.data_path = "data/reuters.json"
  config.num_labels = 3
  config.vocab_path = "data/vocab.txt"
  config.max_seq_len = 128
  run_protonet(config, n_way=5, k_shot=1, n_query=5, 
             n_meta_test_way=5, k_meta_test_shot=10, n_meta_test_query=10,
             logdir="./logs/proto", meta_train=True, 
             n_epochs=2, n_episodes=10, n_meta_test_episodes=1000)