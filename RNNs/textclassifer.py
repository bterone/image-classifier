from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pylot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                            as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

tokenizer = info.features['text'].encoder

print ('Vocabulary size: {}'.format(tokenizer.vocab_size))