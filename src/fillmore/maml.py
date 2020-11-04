"""MAML model code"""
import numpy as np
import sys
import tensorflow as tf
from functools import partial

from tensorflow.python.keras.backend import update
from fillmore.utils import *
from fillmore.bert_model import BertTextClassification
from transformers import BertConfig
    
class MAML(tf.keras.Model):
    def __init__(self, model_config,
                 num_inner_updates=1,
                 inner_update_lr=0.4, 
                 learn_inner_update_lr=False
                 ):
        super(MAML, self).__init__()
        self.inner_update_lr = inner_update_lr
        self.loss_func = cross_entropy_loss
        self.config = model_config

        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
        accuracies_tr_pre, accuracies_ts = [], []

        # for each loop in the inner training loop
        outputs_ts = [[]]*num_inner_updates
        losses_ts_post = [[]]*num_inner_updates
        accuracies_ts = [[]]*num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        tf.random.set_seed(seed)
        self.forward = BertTextClassification(self.config)
        self.forward(self.forward.dummy_text_inputs) #initialize weights
        self.forward.bert.trainable = False # freeze bert layer, only update classifier layer

        # adjustable learning rate
        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for wt in self.forward.classifier.weights:
                key = wt.name
                self.inner_update_lr_dict[key] = [tf.Variable(
                    self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in range(num_inner_updates)]

    def call(self, inp, meta_batch_size=25, num_inner_updates=1):
        def task_inner_loop(inp, reuse=True,
                            meta_batch_size=25, num_inner_updates=1):
            """
              Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
              Args:
                inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
                  labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
                  labels used for evaluating the model after inner updates.
                  Should be shapes:
                    input_tr: [N*K, ]
                    input_ts: [N*K, ]
                    label_tr: [N*K, N]
                    label_ts: [N*K, N]
              Returns:
                task_output: a list of outputs, losses and accuracies at each inner update
            """
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            task_output_tr_pre = self.forward(input_tr)
            task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)
    
            # make a copy of the weights
            weights = self.forward.classifier.weights # has to be initialized first
            fast_weights = weights.copy()
            old_weights_values = self.forward.classifier.get_weights()

            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            #############################

            # Compute accuracies from output predictions
            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1), tf.argmax(
                input=tf.nn.softmax(task_output_tr_pre), axis=1))

            for j in range(num_inner_updates):

                # 1. if j > 1, the gradient tape has to be defined inside the loop
                # 2. as fast_weights are being updated, they should be watched in order to
                # reflect changes to the gradients, i.e., to evaluate the gradients
                # in reference to the updated weights.
                # 3. persistent=True is not required unless the gradients needs to be
                # computed more than once outside the gradient tape
                with tf.GradientTape(watch_accessed_variables=False) as train_tape:
                    train_tape.watch(fast_weights)
                    task_output_tr_pre = self.forward(input_tr, training=True)
                    task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)

                # check if variables are being watched
                # print([var.name for var in train_tape.watched_variables()])

                # will the gradients changes after weights are updated to fast_weights?
                gradients = train_tape.gradient(task_loss_tr_pre, fast_weights)

                # update weights using gradient descent
                # by including inner_update_lr_dict in the computation
                # all the learning rate related variables are being watched by
                # the gradient tape in the outer loop
                fast_weights_values = []
                for i, (wt, grad) in enumerate(zip(fast_weights, gradients)):
                    key = wt.name
                    if self.learn_inner_update_lr:
                        wt = wt - self.inner_update_lr_dict[key][j] * grad
                        fast_weights_values.append(wt.numpy())
                        fast_weights[i].assign(wt)
                    else:
                        wt = wt - self.inner_update_lr * grad # this return a tensor
                        fast_weights_values.append(wt.numpy())
                        fast_weights[i].assign(wt) # reassign to variable
                
                # compute test loss using new weights
                self.forward.classifier.set_weights(fast_weights_values)
                task_output_ts = self.forward(input_ts, training=False)
                task_loss_ts = self.loss_func(task_output_ts, label_ts)

                task_outputs_ts.append(task_output_ts)
                task_losses_ts.append(task_loss_ts)
                task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1), tf.argmax(
                    input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

            # reset the weights
            self.forward.classifier.set_weights(old_weights_values)

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre,
                           task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
        unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                                 False,
                                 meta_batch_size,
                                 num_inner_updates)
        out_dtype = [tf.float32, [tf.float32]*num_inner_updates,
                     tf.float32, [tf.float32]*num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
        task_inner_loop_partial = partial(
            task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial,
                           elems=(input_tr, input_ts, label_tr, label_ts),
                           dtype=out_dtype,
                           parallel_iterations=meta_batch_size)
        return result
