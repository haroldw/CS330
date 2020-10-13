"""MAML model code"""
import numpy as np
import sys
import tensorflow as tf
from functools import partial
from tensorflow.keras import layers
import util

class MAML(tf.keras.Model):
  def __init__(self, dim_input=1,
               dim_output=1,
               num_inner_updates=1,
               inner_update_lr=0.4,
               num_filters=32,
               k_shot=5,
               learn_inner_update_lr=False):
    super(MAML, self).__init__()
    self.dim_input = dim_input
    self.dim_output = dim_output
    self.inner_update_lr = inner_update_lr
    self.loss_func = partial(util.cross_entropy_loss, k_shot=k_shot)
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
    tf.random.set_seed(util.seed)
    self.conv_layers = util.ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)

    self.learn_inner_update_lr = learn_inner_update_lr
    if self.learn_inner_update_lr:
      self.inner_update_lr_dict = {}
      for key in self.conv_layers.conv_weights.keys():
        self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in range(num_inner_updates)]
  

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
      # FIXME: Reshape inputs to N*K

      # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
      weights = self.conv_layers.conv_weights

      # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
      # evaluated on the inner loop training data
      task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

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
      # make a copy of the current weight
      N, K, M = input_tr.shape
      input_tr = tf.reshape(input_tr, (N*K, M))
      input_ts = tf.reshape(input_ts, (N*K, M))
      label_tr = tf.reshape(label_tr, (N*K, N))
      label_ts = tf.reshape(label_ts, (N*K, N))
      gamma = weights.copy()
      for j in range(num_inner_updates):
        with tf.GradientTape(persistent=True) as g:
          task_output_tr_pre = self.conv_layers(input_tr, gamma)
          task_loss_tr_pre = self.loss_func(task_output_tr_pre, label_tr)

          grad = g.gradient(task_loss_tr_pre, weights)
          for name in gamma:
            lr = self.inner_update_lr
            if self.learn_inner_update_lr:
              lr = self.inner_update_lr_dict[name][j]
            gamma[name] = gamma[name] - lr * grad[name]

        pred = self.conv_layers(input_ts, gamma)
        task_outputs_ts.append(pred)
        task_losses_ts.append(self.loss_func(pred, label_ts))
      #############################

      # Compute accuracies from output predictions
      task_accuracy_tr_pre = util.accuracy(tf.argmax(input=label_tr, axis=-1), tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))

      for j in range(num_inner_updates):
        task_accuracies_ts.append(util.accuracy(tf.argmax(input=label_ts, axis=-1), tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

      task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

      return task_output

    input_tr, input_ts, label_tr, label_ts = inp
    # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
    unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                          False,
                          meta_batch_size,
                          num_inner_updates)
    out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32,
                             [tf.float32]*num_inner_updates]
    out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
    task_inner_loop_partial = partial(task_inner_loop,
                                      meta_batch_size=meta_batch_size,
                                      num_inner_updates=num_inner_updates)
    result = tf.map_fn(task_inner_loop_partial,
                       elems=(input_tr, input_ts, label_tr, label_ts),
                       dtype=out_dtype,
                       parallel_iterations=meta_batch_size)
    return result
   
class ProtoNet(tf.keras.Model):

  def __init__(self, num_filters, latent_dim):
    super(ProtoNet, self).__init__()
    self.num_filters = num_filters
    self.latent_dim = latent_dim
    num_filter_list = self.num_filters + [latent_dim]
    self.convs = []
    for i, num_filter in enumerate(num_filter_list):
      block_parts = [
        layers.Conv2D(
          filters=num_filter,
          kernel_size=3,
          padding='SAME',
          activation='linear'),
      ]

      block_parts += [layers.BatchNormalization()]
      block_parts += [layers.Activation('relu')]
      block_parts += [layers.MaxPool2D()]
      block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
      self.__setattr__("conv%d" % i, block)
      self.convs.append(block)
    self.flatten = tf.keras.layers.Flatten()

  def call(self, inp):
    out = inp
    for conv in self.convs:
      out = conv(out)
    out = self.flatten(out)
    return out

def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
  """
    calculates the prototype network loss using the latent representation of x
    and the latent representation of the query set
    Args:
      x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
      q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
      labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
      num_classes: number of classes (N) for classification
      num_support: number of examples (S) in the support set
      num_queries: number of examples (Q) in the query set
    Returns:
      ce_loss: the cross entropy loss between the predicted labels and true labels
      acc: the accuracy of classification on the queries
  """
  #############################
  #### YOUR CODE GOES HERE ####

  # compute the prototypes
  # compute the distance from the prototypes
  # compute cross entropy loss
  # note - additional steps are needed!
  # return the cross-entropy loss and accuracy

  # labels = tf.argmax(labels_onehot, axis=-1)
  x_labels = labels_onehot[:, :, num_support]
  q_labels = labels_onehot[:, :, -num_queries:]
  x_labels = tf.reshape(x_labels, (-1,num_classes))
  q_labels = tf.reshape(q_labels, (-1,num_classes))

  centroids = tf.zeros((num_classes, x_latent.shape[-1]))
  labels = tf.identity(num_classes)
  import pdb
  pdb.set_trace()
  for c in range(num_classes):
    centroids[c,:] = tf.reduce_mean(tf.boolean_mask(x_latent, np.all(x_labels==labels[c],axis=1)),axis=0)

  distances = tf.zeros((num_queries*num_classes, num_classes))

  for i in range(len(q_labels)):
    distances[i] = tf.norm(q_latent[i] - centroids, axis=1)

  ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(q_labels, distances))
  acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(q_labels, axis=-1), tf.argmax(distances, axis=-1)), dtype=tf.float32))

  #############################
  return ce_loss, acc