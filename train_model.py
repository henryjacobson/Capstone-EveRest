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
    def __init__(self, features, labels, onehot, mask):
        self.features = features
        self.labels = labels
        self.onehot = onehot
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
        extras = np.zeros([longest - current.shape[0], n_features], dtype = np.float32)
        features[i] = np.append(current, extras, axis = 0)

        mask_ones = np.ones(current.shape[0], dtype = np.float32)
        mask_zeros = np.zeros(longest - current.shape[0], dtype = np.float32)
        mask.append(np.where(np.append(mask_ones, mask_zeros) > 0.5, True, False))

        labels[i] = np.append(labels[i], mask_zeros)

    features = np.array(features)
    labels = np.array(labels).astype(np.int)
    mask = np.array(mask)

    features = tf.convert_to_tensor(features)
    labels = tf.convert_to_tensor(labels)
    onehot = tf.one_hot(labels, 2)
    mask = tf.convert_to_tensor(mask)
    return Data(features, labels, onehot, mask)

class LSTMModel(Model):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm0 = LSTM(32, return_sequences = True)
        self.lstm1 = LSTM(32, return_sequences = True)
        self.lstm2 = LSTM(32, return_sequences = True)
        self.dense = Dense(2)

    def call(self, x, mask = None):
        x = self.lstm0(x, mask = mask, training = True)
        x = relu(x)
        x = self.lstm1(x, mask = mask, training = True)
        x = relu(x)
        x = self.lstm2(x, mask = mask, training = True)
        x = relu(x)
        x = self.dense(x)
        return x

def train_model(model, n_epochs, data = None):
    @tf.function
    def training_step(features, labels, mask):
        train_accuracy.reset_states()
        with tf.GradientTape() as tape:
            predictions = model(features, mask)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels, predictions, 1000*1000*1000*1000)   #loss_object(labels, predictions, sample_weight = tf.constant([.25, .75]))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy(data.labels, predictions)
        return loss, predictions



    # tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    optimizer = tf.keras.optimizers.SGD()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    if data is None:
        data = load_training_data()

    for epoch in range(n_epochs):
        loss, predictions = training_step(data.features, data.onehot, data.mask)

        asleep_idxs = data.mask#np.bitwise_and(data.labels == 0, data.mask)
        asleep = np.argmin(predictions[asleep_idxs], axis = 1)
        awake_idxs = np.bitwise_and(data.labels == 1, data.mask)
        awake = np.argmax(predictions[awake_idxs], axis = 1)
        awake_class = np.argmax(predictions[data.mask], axis = 1)

        print('Epoch {}, Accuracy {}, asleep {}, awake {}/{}'.format(epoch + 1, train_accuracy.result(), sum(asleep), sum(awake), sum(awake_class)))
