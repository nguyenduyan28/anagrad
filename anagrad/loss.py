import numpy as np

import sys
sys.path.append('../anagrad')
from anagrad.utils import onehot_encoding

class CrossEntropyLoss:
  def softmax(self, X):
    e_max = np.max(X, axis = 1, keepdims=True)
    e_arr = np.exp(X - e_max)
    sum_e_arr = np.sum(e_arr, axis=1, keepdims=True)
    return e_arr / sum_e_arr
  def __init__(self, input=None, target=None):
    self._input = (input) if isinstance(input, np.ndarray) else np.array(input)
    self._target = (target) if isinstance(target, np.ndarray) else np.array(target)
    self._input = (self.softmax(self._input))
    self._target = (onehot_encoding(self._target, self._input.shape[1]))
  def calc(self):
    return np.array(-(np.sum(np.log(self._input) * self._target)/ self._input.shape[0]))
 