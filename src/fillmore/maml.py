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
        dummy_outputs = self.forward(self.forward.dummy_text_inputs) # initialize weights

        # adjustable learning rate
        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for w in self.forward.classifier.weights:
                key = w.name
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
            # fast_weights = self.forward.classifier.weights.copy()
            # not work
            # self.forward.classifier_copy.set_weights(self.forward.classifier.get_weights())
            # fast_weights = self.forward.classifier_copy.weights
            # make copy of model
            forward_copied = BertTextClassification(self.config)
            forward_copied(forward_copied.dummy_text_inputs, training=False)
            forward_copied.set_weights(self.forward.get_weights())
            fast_weights = forward_copied.classifier.weights
            # print([w.name for w in fast_weights])

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            # TODO do we have to evaluate forward pass before inner gradients
            # not sure if persistent is required here, the gradients are computed multiple times
            # it could be defined inside the num_inner_updates loop
            # with tf.GradientTape(persistent=True) as train_tape:
            #   task_output_tr_pre = self.forward(input_tr, fast_weights)
            #   task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            #############################
            #### YOUR CODE GOES HERE ####
            # perform num_inner_updates to get modified weights
            # modified weights should be used to evaluate performance
            # Note that at each inner update, always use input_tr and label_tr for calculating gradients
            # and use input_ts and labels for evaluating performance

            # HINTS: You will need to use tf.GradientTape().
            # Read through the tf.GradientTape() documentation to see how 'persistent' should be set.
            # Here is some documentation that may be useful:
            # https://www.tensorflow.org/guide/advanced_autodiff#higher-order_gradients
            # https://www.tensorflow.org/api_docs/python/tf/GradientTape

            #############################

            # Compute accuracies from output predictions
            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1), tf.argmax(
                input=tf.nn.softmax(task_output_tr_pre), axis=1))
            # fast_weights = weights.copy()

            for j in range(num_inner_updates):

                # 1. if j > 1, the gradient tape has to be defined inside the loop
                # 2. as fast_weights are being updated, they should be watched in order to
                # reflect changes to the gradients, i.e., to evaluate the gradients
                # in reference to the updated weights.
                # 3. persistent=True is not required unless the gradients needs to be
                # computed more than once outside the gradient tape
                with tf.GradientTape(watch_accessed_variables=False) as train_tape:
                    train_tape.watch(fast_weights)
                    task_output_tr_pre = forward_copied(input_tr, training=True)
                    # fast_weights_values = [w.numpy() for w in fast_weights]
                    # task_output_tr_pre = forward_copied(
                    #     input_tr, weights=fast_weights_values, update_copy=False, training=True)
                    # task_output_tr_pre = self.forward(input_tr, weights=fast_weights_values, update_copy=True, training=True)
                    task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)

                # check if variables are being watched
                # print([var.name for var in train_tape.watched_variables()])

                # will the gradients changes after weights are updated to fast_weights?
                gradients = train_tape.gradient(task_loss_tr_pre, fast_weights) #TODO gradients return NONE

                # update weights using gradient descent
                # by including inner_update_lr_dict in the computation
                # all the learning rate related variables are being watched by
                # the gradient tape in the outer loop
                fast_weights_values = []
                for i, (w, grad) in enumerate(zip(fast_weights, gradients)):
                    key = w.name
                    if self.learn_inner_update_lr:
                        w = w - \
                            self.inner_update_lr_dict[key][j] * grad
                        fast_weights_values.append(w.numpy())
                    else:
                        w = w - \
                            self.inner_update_lr * grad
                        fast_weights_values.append(w.numpy())

                # compute test loss using new weights
                task_output_ts = forward_copied(input_ts, training=False)
                task_loss_ts = self.loss_func(task_output_ts, label_ts)

                # gradients = test_tape.gradient(task_loss_ts, weights)
                task_outputs_ts.append(task_output_ts)
                task_losses_ts.append(task_loss_ts)
                task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1), tf.argmax(
                    input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

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
