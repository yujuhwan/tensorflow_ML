# Lab07-3 Application & Tips -> Data & Learning
# IMDB-Text Classification

# 기본 Library 선언 및 TensorFlow 버전 확인
import os
os.environ['TF_CPP_MIN_LOG_LEVEl'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# Tensorflow 2.0 버전에 맞게 Keras를 활용한 IMDB 분류 모델 생성
# 학습에 쓰이는 Data -> 50,000 movie reviews from the Internet Movie Database
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

# IMDB Data를 Vector를 실제 값으로 변환하여 출력
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[4])

print(train_labels[4])

# Tensorflow Keras
# 위 Data를 기준으로 분류 모델을 만듬
# 학습과 평가를 위해 동일길이인 256길이의 단어로 PAD값을 주어 맞춤(뒤의 길이는 0값으로 맞춰줌)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(test_data[0]))
print(train_data[0])

# Tensorflow Keras API를 통해 모델에 대한 정의
# 입력 Size와 학습시킬 Layer의 크기와 Activation Fuction 정의
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# Adam Optimizer과 Cross Entropy Loss 선언
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델을 평가할 Test 데이타에 대한 정의(10000을 기준으로 학습과 평가 수행)
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)
