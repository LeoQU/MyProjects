# -*- coding: utf-8 -*- 

import argparse
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

FLAGS = None


def main(_):

  # 导入数据
  mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

####################################### Create the model #########################################################
  x = tf.placeholder(tf.float32, [None, 784]) # None表示数据的样本量可以是任意值，784是指28 * 28像素的图像
  W = tf.Variable(tf.zeros([784, 10])) # 一般来说参数都放在Variable类型下，因为网络只有一层，因此一共有784个输入，和10个类别的输出
  b = tf.Variable(tf.zeros([10])) # 每个类别的输出都包含一个bias，一般来说一层有多少输出就有多少bias
  y = tf.matmul(x, W) + b # 定义预测模型，这里的y还没有normalized，即还没有代入softmax函数，输出的预测是一个one-hot vector

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10]) # placeholder创建的并不是一个具体的值，而是存储数值的对象，数值将在以后传进去，y_表示真实的label值

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.

  # 使用cross entropy作为loss function，这里logits()方法自动将y和y_normalized，即先代入softmax函数，再计算cross entropy；mean()方法计算平均值
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

  # 采用gradient Descent训练模型，最小话loss function，即cross entropy
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

########################################## end model creation #####################################################

  # 运行模型  
  sess = tf.InteractiveSession() # 启动模型session
  tf.initialize_all_variables().run() # 初始化参数
  # sess.run(tf.initialize_all_variables()) # 也可以这样进行初始化，run()方法可多次调用

  # 进行训练
  for _ in range(1000): # 训练1000次，每次选100个数据
    batch_xs, batch_ys = mnist.train.next_batch(100)   # 每次选100个数据进行训练

    # 使用feed_dict参数将数据输入，train_step参数就是之前定义的梯度下降法最小化cross entropy，注意feed_dict接受字典参数，并将值代入x和y_，输入的值batch_xs和bathc_ys是numpy arrary
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  ### 定义evaluation模型 ###
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # y和y_是二维的numpy array，每行是一个one-hot vector，对应labels，一共有784行对应所有数据，argmax返回每行最大值，equal判断是否相等
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # cast方法将一个tensor中的数据类型转化成某种类型，这里是将True、False转化成float值，即0、1
  ### end ###

  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # 输出准确率，注意，这里调用了新的模型accuracy，并代入新的输入

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
