# -*- coding: utf-8 -*- 

import pickle
import numpy as np
import tensorflow as tf

import random
random.seed(233)

######################
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save

######################
image_size = 28 # pixel * pixel
num_labels = 10 # 10-d one-hot vector labels
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape( (-1, image_size, image_size, num_channels) ).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

######################
with tf.Graph().as_default() as graph:
  stddev = 0.1
  constant = 1.0

  patch_size = 5

  # input dataset
  x  = tf.placeholder( tf.float32, shape=(None, image_size, image_size, num_channels) )
  y_ = tf.placeholder( tf.float32, shape=(None, num_labels) )
  
  # conv layer 1
  depth1 = 32
  layer1_weights = tf.Variable( tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev = stddev) )
  layer1_biases  = tf.Variable( tf.constant(constant, shape=[depth1]) )

  conv = tf.nn.relu( conv2d(x, layer1_weights) + layer1_biases )
  conv = max_pool_2x2(conv)

  # conv layer 2
  depth2 = 64
  layer2_weights = tf.Variable( tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev = stddev) )
  layer2_biases  = tf.Variable( tf.constant(constant, shape=[depth2]) )

  conv = tf.nn.relu( conv2d(conv, layer2_weights) + layer2_biases )
  conv = max_pool_2x2(conv)

  # fully-connected hidden layer
  shape = conv.get_shape().as_list()

  num_hidden = 1024
  layer3_weights = tf.Variable( tf.truncated_normal([shape[1] * shape[2] * shape[3], num_hidden], stddev = stddev) )
  layer3_biases  = tf.Variable( tf.constant(constant, shape=[num_hidden]) )

  reshape = tf.reshape(conv, [-1, shape[1] * shape[2] * shape[3]])
  hidden = tf.nn.relu( tf.matmul(reshape, layer3_weights) + layer3_biases )

  # dropout
  keep_prob = tf.placeholder(tf.float32)
  hidden = tf.nn.dropout(hidden, keep_prob)

  # fully-connected output layer
  layer4_weights = tf.Variable( tf.truncated_normal([num_hidden, num_labels], stddev = stddev) )
  layer4_biases  = tf.Variable( tf.constant(constant, shape=[num_labels]) )

  y = tf.matmul(hidden, layer4_weights) + layer4_biases

  # loss function
  weights = [layer1_weights, layer2_weights, layer3_weights, layer4_weights]

  loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y, y_) )
  # loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y, y_) ) + 0.01 * sum( [tf.nn.l2_loss(i) for i in weights] )
  
  # decaying learning rate
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.01
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500000, 0.96, staircase=True)

  # optimizer
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  # optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
  
  # accuracy
  correct_prediction = tf.equal( tf.argmax(y, 1), tf.argmax(y_, 1) )
  accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )


#####################
num_steps = 1000001
batch_size = 20

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()

  print("Initialized")

  for step in range(num_steps):
    # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    offset = random.randint(0, train_labels.shape[0])
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    session.run( optimizer, feed_dict = {x: batch_data, y_: batch_labels, keep_prob: 0.5} )
    if step % 1000 == 0:
      print( '%d Step Accuracy: %f' % (step, accuracy.eval(feed_dict={x: valid_dataset, y_: valid_labels, keep_prob:1.0})) )

  print( 'Final Accuracy: %f' % accuracy.eval(feed_dict={x: test_dataset, y_: test_labels, keep_prob:1.0}) )
