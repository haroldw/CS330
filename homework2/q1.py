"""Model training code"""
"""
Usage Instructions:
  5-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=25 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1 --logdir=logs/omniglot5way/
  20-way, 1-shot omniglot:
    python main.py --meta_train_iterations=15000 --meta_batch_size=16 --k_shot=1 --n_way=20 --inner_update_lr=0.1 --num_inner_updates=5 --logdir=logs/omniglot20way/
  To run evaluation, use the '--meta_train=False' flag and the '--meta_test_set=True' flag to use the meta-test set.
"""
import numpy as np
import random
import tensorflow as tf
from optparse import OptionParser

from data_generator import DataGenerator
from model import MAML

def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
  with tf.GradientTape(persistent=False) as outer_tape:
    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

  gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
  optim.apply_gradients(zip(gradients, model.trainable_variables))

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

  return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts

def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
  result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

  outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

  return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts  


def meta_train_fn(model, exp_string, data_generator,
               n_way=5, meta_train_iterations=15000, meta_batch_size=25,
               log=True, logdir='/tmp/data', k_shot=1,
               num_inner_updates=1, meta_lr=0.001, inner_lr=0.04):
  SUMMARY_INTERVAL = 10
  SAVE_INTERVAL = 100
  PRINT_INTERVAL = 10  
  TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

  pre_accuracies, post_accuracies = [], []

  num_classes = data_generator.num_classes

  optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)

  # modelName = f'Inner_LR-{inner_lr}'
  writer = tf.summary.create_file_writer(f'./models/{exp_string}/')

  with writer.as_default():
    for itr in range(meta_train_iterations):
      #############################
      #### YOUR CODE GOES HERE ####

      # sample a batch of training data and partition into
      # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
      # NOTE: The code assumes that the support and query sets have the same number of examples.
      images, labels = data_generator.sample_batch(batch_type='meta_train',
                                                  batch_size=meta_batch_size)
      
      K = images.shape[-2]
      support_set_size = int(K/2)
      input_tr = images[:,:,:support_set_size]
      input_ts = images[:,:,-support_set_size:]

      label_tr = labels[:,:,:support_set_size]
      label_ts = labels[:,:,-support_set_size:]
      #############################

      inp = (input_tr, input_ts, label_tr, label_ts)
      
      result = outer_train_step(inp, model, optimizer, 
                                meta_batch_size=meta_batch_size,
                                num_inner_updates=num_inner_updates)

      if itr % SUMMARY_INTERVAL == 0:
        pre_accuracies.append(result[-2])
        post_accuracies.append(result[-1][-1])
        tf.summary.scalar('Train Loss Pre', np.mean(result[2]), step=itr)
        tf.summary.scalar('Train Loss Post', np.mean(result[3]), step=itr)
        writer.flush()

      if (itr!=0) and itr % PRINT_INTERVAL == 0:
        print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies))
        print(print_str)
        tf.summary.scalar('Test Accuracy Pre', np.mean(pre_accuracies), step=itr)
        tf.summary.scalar('Test Accuracy Post', np.mean(post_accuracies), step=itr)
        writer.flush()
        pre_accuracies, post_accuracies = [], []

      if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
        #############################
        #### YOUR CODE GOES HERE ####

        # sample a batch of validation data and partition it into
        # the support/training set (input_tr, label_tr) and the query/test set (input_ts, label_ts)
        # NOTE: The code assumes that the support and query sets have the same number of examples.
        images, labels = data_generator.sample_batch(batch_type='meta_val',
                                                  batch_size=meta_batch_size)
        K = images.shape[-2]
        support_set_size = int(K/2)
        input_tr = images[:,:,:support_set_size]
        input_ts = images[:,:,-support_set_size:]

        label_tr = labels[:,:,:support_set_size]
        label_ts = labels[:,:,-support_set_size:]
        #############################

        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        meta_val_accruracy_pre = result[-2]
        meta_val_accuracy_post = result[-1][-1]
        tf.summary.scalar('Meta-validation Accuracy Pre', meta_val_accruracy_pre, step=itr)
        tf.summary.scalar('Meta-validation Accuracy Post', meta_val_accuracy_post, step=itr)
        writer.flush()
        print('Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (result[-2], result[-1][-1]))

    model_file = logdir + '/' + exp_string +  '/model' + str(itr)
    print("Saving to ", model_file)
    model.save_weights(model_file)

# calculated for omniglot
NUM_META_TEST_POINTS = 600

def meta_test_fn(model, data_generator, n_way=5, meta_batch_size=25, k_shot=1,
              num_inner_updates=1):
  
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
    images, labels = data_generator.sample_batch(batch_type='meta_test',
                                                 batch_size=meta_batch_size)
    K = images.shape[-2]
    support_set_size = int(K/2)
    input_tr = images[:,:,:support_set_size]
    input_ts = images[:,:,-support_set_size:]

    label_tr = labels[:,:,:support_set_size]
    label_ts = labels[:,:,-support_set_size:]
    #############################
    inp = (input_tr, input_ts, label_tr, label_ts)
    result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

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
             data_path='./omniglot_resized',meta_train=True,
             meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1):


  # call data_generator and get data with k_shot*2 samples per class
  data_generator = DataGenerator(n_way, k_shot*2, n_way, k_shot*2, config={'data_folder': data_path})

  # set up MAML model
  dim_output = data_generator.dim_output
  dim_input = data_generator.dim_input
  model = MAML(dim_input,
              dim_output,
              num_inner_updates=num_inner_updates,
              inner_update_lr=inner_update_lr,
              k_shot=k_shot,
              num_filters=num_filters,
              learn_inner_update_lr=learn_inner_update_lr)

  if meta_train_k_shot == -1:
    meta_train_k_shot = k_shot
  if meta_train_inner_update_lr == -1:
    meta_train_inner_update_lr = inner_update_lr

  exp_string = 'cls_'+str(n_way)+'.mbs_'+str(meta_batch_size) + '.k_shot_' + str(meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)


  if meta_train:
    meta_train_fn(model, exp_string, data_generator,
                  n_way, meta_train_iterations, meta_batch_size, log, logdir,
                  k_shot, num_inner_updates, meta_lr)
  else:
    meta_batch_size = 1

    model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
    print("Restoring model weights from ", model_file)
    model.load_weights(model_file)

    meta_test_fn(model, data_generator, n_way, meta_batch_size, k_shot, num_inner_updates)
  
if __name__ == '__main__':
  itr_cnt = 7000
  # run_maml(n_way=5, k_shot=1, num_inner_updates=1, meta_train_iterations=itr_cnt,
  #         inner_update_lr=0.04, learn_inner_update_lr=False)
  # run_maml(n_way=5, k_shot=1, num_inner_updates=1, meta_train_iterations=itr_cnt,
  #          inner_update_lr=0.4, learn_inner_update_lr=False)
  # run_maml(n_way=5, k_shot=1, num_inner_updates=1, meta_train_iterations=itr_cnt,
  #          inner_update_lr=4., learn_inner_update_lr=False)
  # run_maml(n_way=5, k_shot=1, num_inner_updates=1, meta_train_iterations=itr_cnt,
  #          inner_update_lr=0.04, learn_inner_update_lr=True)
  # run_maml(n_way=5, k_shot=1, num_inner_updates=1, meta_train_iterations=itr_cnt,
  #          inner_update_lr=0.4, learn_inner_update_lr=True)
  # run_maml(n_way=5, k_shot=1, num_inner_updates=1, meta_train_iterations=itr_cnt,
  #          inner_update_lr=4., learn_inner_update_lr=True)
  
  # run_maml(n_way=5, k_shot=1, inner_update_lr=4.,
  #          num_inner_updates=1, meta_train=False, meta_train_k_shot=1)
  # run_maml(n_way=5, k_shot=1, inner_update_lr=0.4,
  #          num_inner_updates=1, meta_train=False, meta_train_k_shot=1)
  # run_maml(n_way=5, k_shot=1, inner_update_lr=0.04,
  #          num_inner_updates=1, meta_train=False, meta_train_k_shot=1)
  # run_maml(n_way=5, k_shot=1, inner_update_lr=4., learn_inner_update_lr=True, 
  #          num_inner_updates=1, meta_train=False, meta_train_k_shot=1)
  # run_maml(n_way=5, k_shot=1, inner_update_lr=.4, learn_inner_update_lr=True,
  #          num_inner_updates=1, meta_train=False, meta_train_k_shot=1)
  # run_maml(n_way=5, k_shot=1, inner_update_lr=.04, learn_inner_update_lr=True,
  #          num_inner_updates=1, meta_train=False, meta_train_k_shot=1)
  for k_shot in [4, 6, 8, 10]:
    run_maml(n_way=5, k_shot=k_shot, inner_update_lr=0.4,
            num_inner_updates=1, meta_train=False, meta_train_k_shot=1)
  