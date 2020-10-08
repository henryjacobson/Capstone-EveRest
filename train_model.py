import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.nn import relu

FEATURES = "data/features/"
LABELS   = "data/labels/"

class Data:
    def __init__(self, features, labels, mask):
        self.features = features
        self.labels = labels
        self.mask = mask

# returns masked and well formed lstm input data
def load_training_data(amount = -1):
    # load features
    features = []
    longest = 0
    count = 0
    for filename in os.listdir(FEATURES):
        if amount != -1 and count >= amount:
            break
        current = np.load(FEATURES + filename)
        if current.shape[0] > longest:
            longest = current.shape[0]
        features.append(current)
        count += 1

    # load labels
    count = 0
    labels = []
    for filename in os.listdir(LABELS):
        if amount != -1 and count >= amount:
            break
        labels.append(np.load(LABELS + filename))
        count += 1

    # pad and mask
    mask = []
    n_features = features[0].shape[1]

    for i, current in enumerate(features):
        extras = np.zeros([longest - current.shape[0], n_features])
        features[i] = np.append(current, extras, axis = 0)

        mask_ones = np.ones(current.shape[0])
        mask_zeros = np.zeros(longest - current.shape[0])
        mask.append(np.append(mask_ones, mask_zeros))

        labels[i] = np.append(labels[i], mask_zeros)

    features = tf.convert_to_tensor(features)
    labels = tf.convert_to_tensor(labels)
    mask = tf.convert_to_tensor(mask)
    return Data(features, labels, mask)

class LSTMModel(Model):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(4, return_sequences = True)
        self.dense = Dense(2)

    def call(self, x, mask = None):
        x = self.lstm(x, mask = mask)
        x = relu(x)
        x = self.dense(x)
        return x

def train_model(model, n_epochs, data = None):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    optimizer = tf.keras.optimizers.Adam()

    if data is None:
        data = load_training_data()

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    for epoch in range(n_epochs):
        train_accuracy.reset_states()

        with tf.GradientTape() as tape:
            predictions = model(data.features, data.mask)
            loss = loss_object(data.labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        asleep_idxs = np.bitwise_and(data.labels == 0, data.mask == 1)
        asleep = np.argmin(predictions[asleep_idxs], axis = 1)
        awake_idxs = np.bitwise_and(data.labels == 1, data.mask == 1)
        awake = np.argmax(predictions[awake_idxs], axis = 1)
        train_accuracy(data.labels, predictions)

        print('Epoch {}, Accuracy {}, asleep {}, awake {}'.format(epoch + 1, train_accuracy.result(), sum(asleep), sum(awake)))
