# -*- coding: utf-8 -*- 

import pickle
import numpy as np
import tensorflow as tf

import random
random.seed(233)

########################################
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

########################################
image_size = 28
num_labels = 10
num_channels = 1

def reformat(dataset, labels):
  dataset = dataset.reshape( (-1, image_size, image_size, num_channels) ).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

#######################################
patch_size = 5
depth = 64
num_hidden = 512

with tf.Graph().as_default() as graph:

  x  = tf.placeholder( tf.float32, shape=(None, image_size, image_size, num_channels) )
  y_ = tf.placeholder( tf.float32, shape=(None, num_labels) )

  keep_prob = tf.placeholder(tf.float32)
  
  # learning model
  def model(data):
    # conv layer 1
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))
    conv = tf.nn.relu( tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME') + layer1_biases )
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv layer 2
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))
    conv = tf.nn.relu( tf.nn.conv2d(conv, layer2_weights, [1, 2, 2, 1], padding='SAME') + layer2_biases )
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully-connected hidden layer
    shape = conv.get_shape().as_list() # get the shape of a tensor, and save the shape to a list
    layer3_weights = tf.Variable(tf.truncated_normal([shape[1] * shape[2] * shape[3], num_hidden], stddev=0.1))
    layer3_biases  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    reshape = tf.reshape(conv, [-1, shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    
    # dropout
    hidden = tf.nn.dropout(hidden, keep_prob)

    # fully-connected output layer
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases  = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    return tf.matmul(hidden, layer4_weights) + layer4_biases, [layer1_weights, layer2_weights, layer3_weights, layer4_weights]
  
  # learning model computation
  logits, weights = model(x)

  # loss function
  loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits, y_) )
  # loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits, y_) ) + 0.01 * sum( [tf.nn.l2_loss(i) for i in weights] )

  # learning rate decay
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.01
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000000, 0.96, staircase=True)
    
  # Optimizer
  # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  # optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

  # accuracy
  correct_prediction = tf.equal( tf.argmax(logits, 1), tf.argmax(y_, 1) )
  accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )
  
######################################
num_steps = 1000001
batch_size = 20

with tf.Session(graph=graph) as session:

  tf.initialize_all_variables().run()
  print('Initialized')

  for step in range(num_steps):
    #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    offset = random.randint(0, train_labels.shape[0])
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    session.run( [optimizer, loss], feed_dict = {x: batch_data, y_: batch_labels, keep_prob: 0.5} )

    if step % 1000 == 0:
      print( '%d Step Accuracy: %f' % (step, accuracy.eval(feed_dict={x: valid_dataset, y_: valid_labels, keep_prob:1.0})) )

  print( 'Final Accuracy: %f' % accuracy.eval( feed_dict={x: test_dataset, y_: test_labels, keep_prob: 1.0}) )
  
  with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
    print('model pickled!')
