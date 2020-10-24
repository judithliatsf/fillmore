# run_ProtoNet
# from PIL import Image
# import glob
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
from fillmore.protonet import ProtoNet, ProtoLoss
from fillmore.load_data import DataGenerator

def proto_net_train_step(model, optim, x, q, labels_ph):
  num_classes, num_support, im_height, im_width, channels = x.shape
  num_queries = q.shape[1]
  x = tf.reshape(x, [-1, im_height, im_width, channels])
  q = tf.reshape(q, [-1, im_height, im_width, channels])

  with tf.GradientTape() as tape:
    x_latent = model(x)
    q_latent = model(q)
    ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

  gradients = tape.gradient(ce_loss, model.trainable_variables)
  optim.apply_gradients(zip(gradients, model.trainable_variables))
  return ce_loss, acc

def proto_net_eval(model, x, q, labels_ph):
  num_classes, num_support, im_height, im_width, channels = x.shape
  num_queries = q.shape[1]
  x = tf.reshape(x, [-1, im_height, im_width, channels])
  q = tf.reshape(q, [-1, im_height, im_width, channels])

  x_latent = model(x)
  q_latent = model(q)
  ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)

  return ce_loss, acc 

def run_protonet(data_path='./omniglot_resized', n_way=20, k_shot=1, 
                 n_query=5, n_meta_test_way=20, k_meta_test_shot=5, 
                 n_meta_test_query=5, logdir="./proto", meta_train=True,
                 n_epochs=20, n_episodes=100):
  # n_epochs = 20
  # n_episodes = 100

  im_width, im_height, channels = 28, 28, 1
  num_filters = 32
  latent_dim = 16
  num_conv_layers = 3
  n_meta_test_episodes = 1000
  exp_string = 'cls_'+str(n_way)+'.eps_'+str(n_episodes) + '.k_shot_' + str(k_shot) + '.n_query_' + str(n_query)
  model = ProtoNet([num_filters]*num_conv_layers, latent_dim)
  optimizer = tf.keras.optimizers.Adam()

    # call DataGenerator with k_shot+n_query samples per class
  data_generator = DataGenerator(n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query)
  
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
          images_train, labels_train = data_generator.sample_batch('meta_train', 1, shuffle=False, swap=False)
          images = images_train[0]
          labels = labels_train[0]
          N, K, dim_in = images.shape
          support = tf.reshape(images[:, :k_shot, :], [N, k_shot, im_width, im_height, channels])
          query = tf.reshape(images[:, k_shot:, :], [N, n_query, im_width, im_height, channels])
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
            images_val, labels_val = data_generator.sample_batch('meta_val', 1, shuffle=False, swap=False)
            images = images_val[0]
            labels = labels_val[0]
            N, K, dim_in = images.shape
            support = tf.reshape(images[:, :k_shot, :], [N, k_shot, im_width, im_height, channels])
            query = tf.reshape(images[:, k_shot:, :], [N, n_query, im_width, im_height, channels])
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
    images_test, labels_test = data_generator.sample_batch('meta_test', 1, shuffle=False, swap=False)
    images = images_test[0]
    labels = labels_test[0]
    N, K, dim_in = images.shape
    support = tf.reshape(images[:, :k_meta_test_shot, :], [n_meta_test_way, k_meta_test_shot, im_width, im_height, channels])
    query = tf.reshape(images[:, k_meta_test_shot:, :], [n_meta_test_way, n_meta_test_query, im_width, im_height, channels])
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
  run_protonet('./omniglot_resized/', n_way=5, k_shot=1, n_query=5, 
             n_meta_test_way=5, k_meta_test_shot=10, n_meta_test_query=10,
             logdir="./logs/proto", meta_train=True, n_epochs=2, n_episodes=10)