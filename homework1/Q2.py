import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from Models import MANN
from DataGenerator import DataGenerator

def train_step(images, labels, model, optim, eval=False):
    with tf.GradientTape() as tape:
        predictions = model(images, labels)
        loss = model.loss_function(predictions, labels)
    if not eval:
        gradients = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))
    return predictions, loss


def main(num_classes=5, num_samples=1, meta_batch_size=16, random_seed=1234,
         itr_cnt=25000, cell_count=128, lr=0.001,
         logDir='./logs', modelDir='./models', shuffle=True):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    data_generator = DataGenerator(num_classes, num_samples + 1)

    o = MANN(num_classes, num_samples + 1, cell_count=cell_count)
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    modelName = f'K={num_samples}&N={num_classes}&CellCount={cell_count}&BS={meta_batch_size}'
    writer = tf.summary.create_file_writer(f'{logDir}/{modelName}/')

    with writer.as_default():
      for step in range(itr_cnt):
          i, l = data_generator.sample_batch('train', meta_batch_size, shuffle=shuffle)
          _, ls = train_step(i, l, o, optim)

          if (step + 1) % 100 == 0:
              print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
              i, l = data_generator.sample_batch('test', 100)
              pred, tls = train_step(i, l, o, optim, eval=True)
              pred = tf.reshape(pred, [-1, num_samples + 1, num_classes, num_classes])
              pred = tf.math.argmax(pred[:, -1, :, :], axis=2)
              l = tf.math.argmax(l[:, -1, :, :], axis=2)
              test_accuracy = tf.reduce_mean(tf.cast(tf.math.equal(pred, l), tf.float32)).numpy()

              tf.summary.scalar('Test Accuracy', test_accuracy, step=step)
              tf.summary.scalar('Training Loss', ls.numpy(), step=step)
              tf.summary.scalar('Test Loss', tls.numpy(), step=step)
              writer.flush()
              o.save_weights(f'{modelDir}/{modelName}')
              print("Test Accuracy", test_accuracy,
                    "Train Loss:", ls.numpy(),
                    "Test Loss:", tls.numpy())

itr_cnt = 25000
# Config = (K,N)
# K: num of samples per class
# N: num of classes
configs = [(1,2), (1,3), (1,4), (5,4)]

for config in configs:
  K, N = config
  config_name = f'K={K}&N={N}'
  print(f'Testing {config_name}')
  main(num_classes=N,
       num_samples=K,
       meta_batch_size=64,
       random_seed=1234,
       itr_cnt=itr_cnt,
       logDir='./drive/My Drive/Stanford/CS330/models/question2',
       modelDir='./drive/My Drive/Stanford/CS330/models/question2')