# Lab06-2 Softmax Zoo_classifier-eager
import os
os.environ['TF_CPP_MIN_LOG_LEVEl'] = '2'

import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]  # 마지막빼고 다
y_data = xy[:, -1]  # 마지막만

nb_classes = 7  # 0 ~ 6 클래스

# Make Y data as onehot shape
Y_one_hot = tf.one_hot(y_data.astype(np.int32), nb_classes)

print(x_data.shape, Y_one_hot.shape)

# Weight and bias setting
W = tf.Variable(tf.random.normal((16, nb_classes)), name='weight')
b = tf.Variable(tf.random.normal((nb_classes,)), name='bias')
variables = [W, b]  # update 된 값


# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def logit_fn(X):
    return tf.matmul(X, W) + b


def hypothesis(X):  # 정확도 업
    return tf.nn.softmax(logit_fn(X))


def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.keras.losses.categorical_crossentropy(y_true=Y, y_pred=logits,
                                                      from_logits=True)
    cost = tf.reduce_mean(cost_i)
    return cost


def grad_fn(X, Y):
    with tf.GradientTape() as tape:  # 최저점
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)  # w, b
        return grads


def prediction(X, Y):  # 정확도 올리기기
    pred = tf.argmax(hypothesis(X), 1) # 제일 큰값 찾기
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def fit(X, Y, epochs=1000, verbose=100):
    optimizer =  tf.keras.optimizers.SGD(learning_rate=0.1)  # 최적화

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i==0) | ((i+1)%verbose==0):
#             print('Loss at epoch %d: %f' %(i+1, cost_fn(X, Y).numpy()))
            acc = prediction(X, Y).numpy()
            loss = cost_fn(X, Y).numpy()
            print('Steps: {} Loss: {}, Acc: {}'.format(i+1, loss, acc))

fit(x_data, Y_one_hot)