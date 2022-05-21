# Lab 07-2 Overfitting

# 기본 Library 선언 및 TensorFlow 버전 확인
import os
os.environ['TF_CPP_MIN_LOG_LEVEl'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(777)  # for reproducibility

print(tf.__version__)

# 정규화를 위한 함수 ( 최대 최소값이 1과 0이 되도록 Scaling한다.)
def normalization(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / denominator

# X Data(feature)의 값은 해당 배열의 첫번째 값부터 4번째 값까지로 정의되고 Y Data(label)
# 은 해당 벼열의 마지막 값을 정의(5번째 값)
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# x_train = xy[:, 0:-1]  # x의 값: 마지막 빼고
# y_train = xy[:, [-1]]  # y의 값: 마지막 것만
#
# plt.plot(x_train, 'ro')
# plt.plot(y_train)
# plt.show()

# Data에 표준화를 적용하여 실행
xy = normalization(xy)
# print(xy)
x_train = xy[:, 0:-1]  # 마지막 제외한 모두
y_train = xy[:, [-1]]  # 마지막 것만

plt.plot(x_train, 'ro')
plt.plot(y_train)

plt.show()

# 위를 기준으로 Linear Regression 모델을 만듬
# Tensorflow data API를 통해 학습시킬 값들을 담는다(Batch Size는 한번에 학습시킬 Size로 정한다)
# X(features),Y(labels)는 실제 학습에 쓰일 Data(연산을 위해 Type을 맞춰준다)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

# W와 b는 학습을 통해 생성되는 모델에 쓰이는 Weight와 Bias(초기값을 variable:0 이나 Random값으로 가능
W = tf.Variable(tf.random.normal((4, 1)), dtype=tf.float32)
b = tf.Variable(tf.random.normal((1,)), dtype=tf.float32)

# Linear Regression의 Hyphthesis를 정의 (y=Wx+b)
def linearReg_fn(features):
    hypothesis = tf.matmul(features, W) + b  # x * w + b
    return hypothesis

# L2 loss를 적용할 함수를 정의
# Weight의 수가 많아지면 수만큼 더한다. / overfitting 문제 해결
def l2_loss(loss, beta = 0.01):
    W_reg = tf.nn.l2_loss(W)  # output = sum(t ** 2) / 2, 기본 l2 loss값의 regulation된 값 적용
    loss = tf.reduce_mean(loss + W_reg * beta)  # 정규화된 값 * beta = 실제 loss에서 정규화된 l2_loss값
    return loss


# 가설을 검증항 Cost함수를 정의(Mean Square Error를 사용)
def loss_fn(hypothesis, features, labels, flag = False):
    cost = tf.reduce_mean(tf.square(hypothesis - labels))  # 가설 - y값의 최소화 = cost
    if(flag):
        cost = l2_loss(cost)
    return cost

# Learning Rate값을 조정하기 위한 Learning Decay 설정
# 5개의 파라미터 설정(starter_learning_rate, global_step, 1000, 0.96, decayed_learning_rate)
is_decay = True
starter_learning_rate = 0.1  # learning rate 값

if(is_decay):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=starter_learning_rate,
                                                                  decay_steps=50,
                                                                  decay_rate=0.96,
                                                                  staircase=True)  # 최초 학습시 사용될 learning rate(0.1로 설정하여 0.96씩 감소하는지 확인)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)  # 최적의 learning rate값

def grad(hypothesis, features, labels, l2_flag):  # gradinet
    with tf.GradientTape() as tape:
        loss_value = loss_fn(linearReg_fn(features),features,labels, l2_flag)
    return tape.gradient(loss_value, [W,b]), loss_value

# TensofFlow를 통해 학습
EPOCHS = 101

for step in range(EPOCHS):  # 학습
    for features, labels  in dataset:
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        grads, loss_value = grad(linearReg_fn(features), features, labels, False)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
    if step % 10 == 0:
        print("Iter: {}, Loss: {:.4f}".format(step, loss_value))