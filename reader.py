import os
import pickle
from collections import Counter
import tensorflow as tf

def counts():
  cache = 'counter.pickle'
  if os.path.exists(cache):
    with open(cache, 'r') as f:
      return pickle.load(f)

  directories = ['cnn/questions/training/', 'cnn/questions/validation/', 'cnn/questions/test/']
  files = [directory + file_name for directory in directories for file_name in os.listdir(directory)]
  counter = Counter()
  for file_name in files:
    with open(file_name, 'r') as f:
      lines = f.readlines()
      document = lines[2].split()
      query = lines[4].split()
      answer = lines[6].split()
      for token in document + query + answer:
        counter[token] += 1
  with open(cache, 'w') as f:
    pickle.dump(counter, f)

  return counter

def tokenize(index, word):
  directories = ['cnn/questions/training/', 'cnn/questions/validation/', 'cnn/questions/test/']
  for directory in directories:
    out_name = directory.split('/')[-2] + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(out_name)
    files = map(lambda file_name: directory + file_name, os.listdir(directory))
    for file_name in files:
      with open(file_name, 'r') as f:
        lines = f.readlines()
        document = [index[token] for token in lines[2].split()]
        query = [index[token] for token in lines[4].split()]
        answer = [index[token] for token in lines[6].split()]
        example = tf.train.Example(
           features = tf.train.Features(
             feature = {
               'document': tf.train.Feature(
                 int64_list=tf.train.Int64List(value=document)),
               'query': tf.train.Feature(
                 int64_list=tf.train.Int64List(value=query)),
               'answer': tf.train.Feature(
                 int64_list=tf.train.Int64List(value=answer))
               }))

      serialized = example.SerializeToString()
      writer.write(serialized)

def main():
  counter = counts()
  print('num words',len(counter))
  word, _ = zip(*counter.most_common())
  index = {token: i for i, token in enumerate(word)}
  tokenize(index, word)
  print('DONE')

if __name__ == "__main__":
  main()
