# Cost function in pure Python

# import numpy as np
#
# X = np.array([1,2,3])
# Y = np.array([1,2,3])
#
# def cost_func(W, X, Y):  # cost function
#     c = 0
#     for i in range(len(X)):
#         c += (W * X[i] - Y[i]) ** 2  # Hypothesis(예측값): W * X[i], 실제값: Y[i], 오차값: W * X[i] - Y[i]
#     return c / len(X)  # 평균
#
# for feed_W in np.linspace(-3, 5, num=15):  # -3 ~ 5까지 15개 구간으로 나눔
#     curr_cost = cost_func(feed_W, X, Y)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

# Cost function in TensorFlow
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEl'] = '2'
#
# import numpy as np
# import tensorflow as tf
#
# X = np.array([1,2,3])
# Y = np.array([1,2,3])
#
# def cost_func(W, X, Y):  # cost function
#    hypothesis = X * W
#    return tf.reduce_mean(tf.square(hypothesis - Y))  # tf.reduce_mean(): 평균, tf.squre(): 제곱
#
# W_values = np.linspace(-3, 5, num=15)  # -3 ~ 5까지 15개 구간으로 나눔
# cost_value = []  # 값을 리스트로
#
# for feed_W in W_values:
#     curr_cost = cost_func(feed_W, X, Y)
#     cost_value.append(curr_cost)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))


# Gradient descent 를 tensorflow롤 나타내기 - > cost 미분
import os
os.environ['TF_CPP_MIN_LOG_LEVEl'] = '2'

import tensorflow as tf
import numpy as np

tf.random.set_seed(0)
x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random.normal([1], -100., 100.))

for step in range(300):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
    descent = W * tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))

