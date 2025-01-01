import tarfile
import numpy as np
import os
import gzip
import pickle
from matplotlib import pyplot as plt
import tempfile
from sklearn.model_selection import train_test_split


# normalize [0, 255] --> [0, 1]

def greyscale_img(img_data):
  img_data_split = np.array(np.split(img_data, 3, axis=1))
  normalize_arr = (np.mean(img_data_split, axis=0, keepdims=True) / 255).reshape(img_data_split.shape[1], img_data_split.shape[2])
  return np.array(normalize_arr)


def onehot_encoding(labels, num_of_classifier = 9):
  labels_onehot = np.zeros((labels.shape[0], num_of_classifier))
  # for i in range(labels.shape[0]):
  #   labels_onehot[i, labels[i]] = 1
  # optimize version
  labels_onehot[np.arange(labels.shape[0]), labels] = 1
  return labels_onehot

def add_ones(X):
  ones = np.ones((X.shape[0], 1))
  return np.concatenate((ones, X), axis=1)



def unpickle(file): # ref: https://www.cs.toronto.edu/~kriz/cifar.html
  train_data = []
  train_label = []
  test_data = []
  test_label = []
  with tarfile.open(file, 'r:gz') as t:
    t.extractall('./data')
    is_test_file = False

    for member in t.getnames():
      if ('test_batch' in member):
        is_test_file = True

      filename = os.path.join('.','data', member)
      try:
        with open(filename, 'rb') as fo:
          batch = pickle.load(fo, encoding='latin1')
          data_file, labels_file = batch['data'], batch['labels']
          if (is_test_file == False):
            train_data.append(data_file)
            train_label.append(labels_file)
          else :
            test_data.append(data_file)
            test_label.append(labels_file)
            is_test_file = False
      except:
        print("Error not data files")
  return np.concatenate(train_data), np.concatenate(test_data), np.concatenate(train_label), np.concatenate(test_label)


def activation_function(X, name = ['Relu', 'Softmax', 'Sigmoid']):
  if (name == 'Relu'):
    X[np.argwhere(X < 0)] = 0
    return X
  if (name == 'Softmax'):
    e_X = np.exp(X) / np.sum(np.exp(X), keepdims= True)
    return e_X
  if (name == 'Sigmoid'):
    return 1 / (1 + np.exp(-X))
  

  






