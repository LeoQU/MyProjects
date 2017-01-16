# -*- coding: utf-8 -*- 

import pickle
import numpy as np
import tensorflow as tf


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

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
 
######################
num_layer_1 = 256 # number of neures in the hidden layer 1

with tf.Graph().as_default() as graph:

  # input set
  x = tf.placeholder( tf.float32, shape=(None, image_size * image_size) )
  y_ = tf.placeholder( tf.float32, shape=(None, num_labels) )
  
  # weights and biases in the hidden layer 1
  weights_hidden = tf.Variable( tf.truncated_normal([image_size * image_size, num_layer_1]) )
  biases_hidden = tf.Variable( tf.constant(0.1, shape=[num_layer_1]) )

  # weights and biases in the final layer
  weights_final = tf.Variable( tf.truncated_normal([num_layer_1, num_labels]) )
  biases_final = tf.Variable( tf.constant(0.1, shape=[num_labels]) )
  
  # computation process
  hidden_layer_1 = tf.nn.relu( tf.matmul(x, weights_hidden) + biases_hidden )
  y = tf.matmul(hidden_layer_1, weights_final) + biases_final

  # loss function
  loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y, y_) ) + 0.01 * tf.nn.l2_loss(weights_hidden) + 0.01 * tf.nn.l2_loss(weights_final)
  
  # decaying learning rate
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 50000, 0.96, staircase=True)

  # optimizer
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # accuracy
  correct_prediction = tf.equal( tf.argmax(y, 1), tf.argmax(y_, 1) )
  accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )


#####################
num_steps = 50001
batch_size = 200

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()

  print("Initialized")

  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    session.run( optimizer, feed_dict = {x: batch_data, y_: batch_labels} )
    if step % 1000 == 0:
      print( '%d Step Accuracy: %f' % (step, accuracy.eval(feed_dict={x: valid_dataset, y_: valid_labels})) )

  print( 'Final Accuracy: %f' % accuracy.eval(feed_dict={x: test_dataset, y_: test_labels}) )
