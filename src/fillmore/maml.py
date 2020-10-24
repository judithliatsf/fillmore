"""MAML model code"""
import numpy as np
import sys
import tensorflow as tf
from functools import partial
from fillmore.utils import *

class MAML(tf.keras.Model):
    def __init__(self, dim_input=1, dim_output=1,
                 num_inner_updates=1,
                 inner_update_lr=0.4, num_filters=32, k_shot=5, learn_inner_update_lr=False):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.loss_func = partial(cross_entropy_loss, k_shot=k_shot)
        self.dim_hidden = num_filters
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input/self.channels))

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
        self.conv_layers = ConvLayers(
            self.channels, self.dim_hidden, self.dim_output, self.img_size)

        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for key in self.conv_layers.conv_weights.keys():
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
                    input_tr: [N*K, 784]
                    input_ts: [N*K, 784]
                    label_tr: [N*K, N]
                    label_ts: [N*K, N]
              Returns:
                task_output: a list of outputs, losses and accuracies at each inner update
            """
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            weights = self.conv_layers.conv_weights
            task_output_tr_pre = self.conv_layers(input_tr, weights)
            task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)
            fast_weights = weights.copy()

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            # TODO do we have to evlaute forward pass before inner gradients
            # not sure if persistent is required here, the gradients are computed multiple times
            # it could be defined inside the num_inner_updates loop
            # with tf.GradientTape(persistent=True) as train_tape:
            #   task_output_tr_pre = self.conv_layers(input_tr, fast_weights)
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
                with tf.GradientTape() as train_tape:
                    variables = list(fast_weights.values())
                    train_tape.watch(variables)
                    task_output_tr_pre = self.conv_layers(
                        input_tr, fast_weights)
                    task_loss_tr_pre = self.loss_func(
                        task_output_tr_pre, label_tr)

                # check if variables are being watched
                # print([var.name for var in train_tape.watched_variables()])

                # will the gradients changes after weights are updated to fast_weights?
                gradients = train_tape.gradient(task_loss_tr_pre, fast_weights)

                # update weights using gradient descent
                # by including inner_update_lr_dict in the computation
                # all the learning rate related variables are being watched by
                # the gradient tape in the outer loop
                for key, v in fast_weights.items():
                    if self.learn_inner_update_lr:
                        fast_weights[key] = v - \
                            self.inner_update_lr_dict[key][j] * gradients[key]
                    else:
                        fast_weights[key] = v - \
                            self.inner_update_lr * gradients[key]

                # compute test loss using new weights
                task_output_ts = self.conv_layers(input_ts, fast_weights)
                # /tf.cast(meta_batch_size, tf.float32)
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
