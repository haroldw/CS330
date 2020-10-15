# run_ProtoNet
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import glob
from data_generator import DataGenerator
from model import ProtoNet, ProtoLoss
import matplotlib.pyplot as plt

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

def run_protonet(data_path='./omniglot_resized', n_way=20, k_shot=1, n_query=5, n_meta_test_way=20, k_meta_test_shot=5, n_meta_test_query=5):
  n_epochs = 20
  n_episodes = 100

  im_width, im_height, channels = 28, 28, 1
  num_filters = 32
  latent_dim = 16
  num_conv_layers = 3
  n_meta_test_episodes = 1000

  model = ProtoNet([num_filters]*num_conv_layers, latent_dim)
  optimizer = tf.keras.optimizers.Adam()

  writer = tf.summary.create_file_writer(f'./models/protonet/')

  with writer.as_default():
    # call DataGenerator with k_shot+n_query samples per class
    data_generator = DataGenerator(n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query)
    for ep in range(n_epochs):
      for epi in range(n_episodes):
        #############################
        #### YOUR CODE GOES HERE ####

        # sample a batch of training data and partition it into
        # support and query sets

        images, labels = data_generator.sample_batch(batch_type='meta_train',
                                                    batch_size=1,
                                                    shuffle=False,
                                                    swap=False)
        images = images.reshape(n_way,k_shot+n_query, 28, 28, 1)
        support = images[:,:k_shot]
        query = images[:,-n_query:]
        #############################
        ls, ac = proto_net_train_step(model, optimizer, x=support, q=query, labels_ph=labels)
        if (epi+1) % 50 == 0:
          #############################
          #### YOUR CODE GOES HERE ####

          # sample a batch of validation data and partition it into
          # support and query sets
          images, labels = data_generator.sample_batch(batch_type='meta_val',
                                                      batch_size=1,
                                                      shuffle=False,
                                                      swap=False)
          images = images.reshape(n_way,k_shot+n_query, 28, 28, 1)
          support = images[:,:k_shot]
          query = images[:,-n_query:]
          #############################
          val_ls, val_ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)
          print('[epoch {}/{}, episode {}/{}] => meta-training loss: {:.5f}, meta-training acc: {:.5f}, meta-val loss: {:.5f}, meta-val acc: {:.5f}'.format(ep+1,
                                                                      n_epochs,
                                                                      epi+1,
                                                                      n_episodes,
                                                                      ls,
                                                                      ac,
                                                                      val_ls,
                                                                      val_ac))
          tf.summary.scalar('Validation Loss', val_ls, step=ep*n_episodes+epi)
          tf.summary.scalar('Validation Acc', val_ac, step=ep*n_episodes+epi)
          writer.flush()

  print('Testing...')
  meta_test_accuracies = []
  for epi in range(n_meta_test_episodes):
    #############################
    #### YOUR CODE GOES HERE ####

    # sample a batch of test data and partition it into
    # support and query sets
    images, labels = data_generator.sample_batch(batch_type='meta_test',
                                                 batch_size=1,
                                                 shuffle=False,
                                                 swap=False)
    images = images.reshape(n_way,k_meta_test_shot+n_meta_test_query, 28, 28, 1)
    support = images[:,:k_meta_test_shot]
    query = images[:,-n_meta_test_query:]
    #############################
    ls, ac = proto_net_eval(model, x=support, q=query, labels_ph=labels)
    meta_test_accuracies.append(ac)
    if (epi+1) % 50 == 0:
      print('[meta-test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_meta_test_episodes, ls, ac))
  avg_acc = np.mean(meta_test_accuracies)
  stds = np.std(meta_test_accuracies)
  print('Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))

if __name__ == '__main__':
  run_protonet('./omniglot_resized/', n_way=5, k_shot=1, n_query=5, n_meta_test_way=5, k_meta_test_shot=4, n_meta_test_query=4)