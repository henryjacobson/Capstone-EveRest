import os
import numpy as np
import tensorflow as tf

FEATURES = "data/features/"
LABELS   = "data/labels/"

class Data:
    def __init__(self, features, labels, mask):
        self.features = features
        self.labels = labels
        self.mask = mask

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
