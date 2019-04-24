import csv
import numpy as np
import tensorflow as tf

reader = csv.reader(open("data.csv", "rb"), delimiter=",")
arr = list(reader)
result = np.array(arr)

X = map(lambda x: x[0], result)
Y = map(lambda x: x[1], result)

X_train, X_test = tf.split(X, [8999, 1000])
Y_train, Y_test = tf.split(Y, [8999, 1000])

tf.shape(X_train)
tf.shape(Y_train)

