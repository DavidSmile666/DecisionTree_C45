import numpy as np
import pandas as pd 
import math

class TreeNode:
  def __init__(self, samples, target, min_sample_split = 2, max_depth = 20):
    self.decision = None            # undecided
    self.samples = samples          # attribute/data X
    self.target = target            # label/target Y
    self.split_attribute = None     # splitting attribute
    self.split_threshold = None          # splitting attribute value
    self.max_depth = max_depth                 # max depth to split a node 
    self.min_sample_split = min_sample_split   # min number of samples to split a node
    self.children = None

  def make(self, cur_depth):
    target = self.target
    samples = self.samples

    if len(target.unique()) == 1:                   # only one class in the target
      self.decision = target.unique()[0]

    elif len(samples) < self.min_sample_split:      # number of sample is less than min_sample_split
      self.decision = target.value_counts().keys()[0]

    elif cur_depth == self.max_depth:               # reach max depth
      self.decision = target.value_counts().keys()[0]

    else:                                           # split a node      
      best_attribute, best_threshold, best_idx = self.split()
      self.split_attribute = best_attribute
      self.split_threshold = best_threshold

      less_sample, geq_sample = samples[best_idx], samples[best_idx == False]
      less_target, geq_target = target[best_idx], target[best_idx == False]

      less_child = TreeNode(less_sample, less_target)
      if len(less_sample) > 0:
        less_child.make(cur_depth + 1)
      else:
        less_child.decision = target.value_counts().keys()[0]
      
      geq_child = TreeNode(geq_sample, geq_target)
      if len(geq_sample) > 0:
        geq_child.make(cur_depth + 1)
      else:
        geq_child.decision = target.value_counts().keys()[0]
      self.children = {"less": less_child, "geq": geq_child}

  def split(self):
    samples = self.samples
    target = self.target

    info_gain_max = -1*float("inf")
    best_attribute = None
    best_threshold = None
    best_idx = None

    for attr in samples.keys():
      sorted_index = samples[attr].sort_values().index
      sorted_attr = samples[attr][sorted_index]
      for i in range(0, len(sorted_attr)-1):
        if sorted_attr.iloc[i] != sorted_attr.iloc[i+1]:
          threshold = (sorted_attr.iloc[i] + sorted_attr.iloc[i+1])/2
          idx = samples[attr] < threshold
          less_target = target[idx]
          geqtarget = target[idx == False]
          ig = self.compute_info_gain(less_target, geq_target, target)

          if ig > info_gain_max:
            info_gain_max = ig
            best_attribute = attr
            best_threshold = threshold
            best_idx = idx

    return (best_attribute, best_threshold, best_idx)

  def compute_entropy(self, target):
    if len(target) < 2:
      return 0
    else:
      freq = np.array(target.value_counts(normalize = True))
      return -(freq * np.log2(freq + 1e-6)).sum()

  def compute_info_gain(self, less_target, geq_target, target):
    p1 = len(less_target)/len(target)
    p2 = len(geq_target)/len(target)

    cond_ent = p1*self.compute_entropy(less_target) + p2*self.compute_entropy(geq_target)
    ig = self.compute_entropy(target) - cond_ent

    ent = - (p1*np.log2(p1 + 1e-6) + p2*np.log2(p2+1e-6))
    return ig/ent

  def predict(self, sample):
    if self.decision is not None:             # reach leaf node
      return self.decision
    else:
      attr_val = sample[self.split_attribute]
      if attr_val < self.split_threshold:
        child = self.children["less"]
      else:
        child = self.children["geq"]
      return child.predict(sample)


class TreeC45:
  def __init__(self):
    self.root = None

  def fit(self, samples, target):
    self.root = TreeNode(samples, target)
    self.root.make(1)

  def predict(self, samples):
    return pd.Series([self.root.predict(samples.iloc[i]) for i in range(0, len(samples))])

  def accuracy(self, truth_target, pred_target):
    acc = truth_target[truth_target == pred_target].count()
    return acc/len(truth_target)


############## main #######################
iris = pd.read_csv("iris.csv")
iris = iris.sample(frac=1).reset_index(drop=True)    # shuffle the data

samples=iris[["sepal length", "sepal width", "petal length", "petal width"]]   # extract the attributes and target
target=iris["class"]

train_sample,test_sample = samples[0:120], samples[120:]         # 80% for training, 20% for test
train_target, test_target = target[0:120], target[120:]
test_sample.index = np.arange(0, len(test_sample))
test_target.index = np.arange(0, len(test_target))

c45 = TreeC45()
c45.fit(train_sample, train_target)               # construct the decision tree

pred_train_target = c45.predict(train_sample)     # measure training error           
acc = c45.accuracy(train_target, pred_train_target)
print("Accuray on training data {:.2f}".format(acc))

pred_test_target = c45.predict(test_sample)       # measure test error
acc = c45.accuracy(test_target, pred_test_target)
print("Accuracy on test data {:.2f}".format(acc))