import re
import pdb
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras

def to_int(arr):
    result = np.array(list(map(lambda val: int(val), arr)))
    size = 8 - result.size
    return np.concatenate((np.zeros(size), result))

def convert_labels_to_integers(labels):
    labels_dict = {
        'T': 0,
        'R': 1,
        'W': 2,
        'A': 3,
        'G': 4,
        'M': 5,
        'Y': 6,
        'F': 7,
        'P': 8,
        'D': 9,
        'X': 10,
        'B': 11,
        'N': 12,
        'J': 13,
        'Z': 14,
        'S': 15,
        'Q': 16,
        'V': 17,
        'H': 18,
        'L': 19,
        'C': 20,
        'K': 21,
        'E': 22
    }

    return list(map(lambda x: labels_dict[x], labels))

# Import data
reader = csv.reader(open("data.csv", "r"), delimiter=",")
result = np.array(list(reader))

# Refactor data
X = np.array(list(map(lambda x: keras.utils.to_categorical(to_int(re.findall(r'\d', x[0])), num_classes=10), result)))
Y = np.array(convert_labels_to_integers(result[:,1]))

# Extract train and test sets
X_train = X[:9000, :, :]
X_test = X[9001:, :, :]
Y_train = Y[:9000]
Y_test = Y[9001:]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8, 10)),
    keras.layers.Dense(184, activation='relu'),
    keras.layers.Dense(92, activation='relu'),
    keras.layers.Dense(46, activation='relu'),
    keras.layers.Dense(23, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=60)
results = model.evaluate(X_test, Y_test)

print(results)

