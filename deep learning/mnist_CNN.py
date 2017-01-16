# -*- coding: utf-8 -*- 

import argparse
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

FLAGS = None

####################################### Help functions ###########################################################
# 设置权重初值
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # Outputs random values from a truncated normal distribution, mean默认为0, 方差为0.1
  return tf.Variable(initial) # 注意，这里依然返回了Variable对象

# 设置bias初值
def bias_variable(shape): 
  initial = tf.constant(0.1, shape=shape) # Creates a constant tensor, 生成内部所有值都是0.1的tensor
  return tf.Variable(initial) # 注意，这里依然返回了Variable对象

# 设置卷积规则
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # 补0， 这里strides=[1, 1, 1, 1]，表示手电筒在input的四个维度上各移动1步，四个维度是[batch, in_height, in_width, in_channels]

# 设置pooling规则
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 这里pool进行了补0，而且步长在height和witda上移动了2步
  # 特别注意，由于在进行pool时，没有weight(W)输入，所以在使用max_pool时就要确定在多大范围内进行max，这里ksize参数就是在4个维度上确定在多大范围上进行max，这里是height * width = 2 * 2
######################################## End #####################################################################

### 主函数 ###
def main(_):

  # 导入数据
  mnist = read_data_sets(FLAGS.data_dir, one_hot=True)
  print( FLAGS.data_dir )

  ####################################### Create the model #########################################################
  # 输入数据
  x = tf.placeholder(tf.float32, [None, 784]) # None表示数据的样本量可以是任意值，784是指28 * 28像素的图像
  
  # 真实的labels
  y_ = tf.placeholder(tf.float32, [None, 10]) # placeholder创建的并不是一个具体的值，而是存储数值的对象，数值将在以后传进去，y_表示真实的label值

  # reshape input x in 4-D，这里改为4维是为了将x输入tf.nn.conv2d，该函数接受4维数据
  x_image = tf.reshape(x, [-1,28,28,1]) # -1表示在进行reshape时，保证28*28*1的维度的情况下，第一维度是多少都可以；这里-1 * 28 * 28 * 1 = number of images * height * width * depth

  ### 第一个卷积pool层对儿 ###
  W_conv1 = weight_variable([5, 5, 1, 32]) # filter size（手电筒的范围）: height - 5, width - 5, input depth - 1 (例如，有三种颜色), output depth - 32 (即人为设置卷积了32层，即用32个手电照墙壁)
  b_conv1 = bias_variable([32]) # 由于有32个手电找墙壁，所以有32个bias

  h_conv1 = tf.nn.relu( conv2d(x_image, W_conv1) + b_conv1 ) # conv2d返回的是x * W的结果，并加上biases，然后输入到RELU函数
  h_pool1 = max_pool_2x2(h_conv1) # 在2*2的范围内求max

  ### 第二个卷积pool层对儿 ###
  W_conv2 = weight_variable([5, 5, 32, 64]) # 和第一层类似，光照面积是5*5，不同的是第一层有32层卷积，第二层有64层卷积，即64个手电筒
  b_conv2 = bias_variable([64])
  # 注意，weight_variable和bias_variable函数的输入都是list，因为要把shape参数整体传进去

  h_conv2 = tf.nn.relu( conv2d(h_pool1, W_conv2) + b_conv2 )
  h_pool2 = max_pool_2x2(h_conv2)

  ### 最后的全连通的hidden层 ###
  W_fc1 = weight_variable([7*7*64, 1024]) # 这时图像变成 7 * 7的平面组合成的深度是64的数据，该数据全连通到一个包含1024个神经元的层
  b_fc1 = bias_variable([1024]) # 由于这一层有1024个神经元，所以需要1024个biases
  # 特别注意，经过两次卷积和两次pool，图像只剩7*7*64的大小，这里7是通过公式：（输入边长 - 光照宽度 + 2 * 补0宽度） / 步长 + 1，计算得出的
  
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # 再将数据重新变回标准的ANN所需的format
  h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W_fc1) + b_fc1 ) # 这里依然使用了RELU函数，使用matmul方法进行矩阵乘法

  ### Dropout ###
  keep_prob = tf.placeholder(tf.float32) # keep_prob应该是一个单个的概率值，而且必须是个tensor
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # keep_prob是dropout的概率，即让多少数据通过，这里expected sum of input并没有变，这是由于dropout方法的自动调整
  # keep_prob is a scalar Tensor with the same type as input x (h_fc1). The probability that each element is kept.
  # With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0. The scaling is so that the expected sum is unchanged.

  ### 最后的全连通输出层 ###
  W_fc2 = weight_variable([1024, 10]) # 输出层有10个类别
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # 注意，这里没有使用RELU函数

  ### 设置loss function ###
  cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y_conv, y_) ) # cross entropy
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 这里不使用gradient descent，而使用ADAM optimizer来进行训练

  ### 设置判断准确率的方法 ###
  correct_prediction = tf.equal( tf.argmax( y_conv,1), tf.argmax(y_,1) )
  accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )
  ################################## End Model Creation ###########################################################

  # 创建CNN网络session
  sess = tf.InteractiveSession()

  # 初始化变量
  # sess.run(tf.initialize_all_variables())
  tf.initialize_all_variables().run() # 也可以使用这种形式

  # 训练
  for i in range(1001): # 进行20000次训练
    batch = mnist.train.next_batch(50) # 每次选50个数据进行训练

    if i%100 == 0: # 每100次训练后，输出log，显示当前的训练准确率
      train_accuracy = accuracy.eval( feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0} )
      print("step %d, training accuracy %g" % (i, train_accuracy))

    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # 注意，这里keep_prob也作为字典参数传了进去
    sess.run( train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5} ) # 也可以使用这种方法运行

  # 输出准确率
  print( accuracy.eval(feed_dict={x: mnist.test.images[0:1000,:], y_: mnist.test.labels[0:1000,:], keep_prob: 1.0}) ) # accuracy.eval() is a shortcut for calling tf.get_default_session().run(accuracy)

### 主函数调用 ####
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data', help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
