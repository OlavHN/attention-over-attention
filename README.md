# Attention over Attention

Implementation of the paper [Attention-over-Attention Neural Networks for Reading Comprehension](https://arxiv.org/abs/1607.04423) in tensorflow

Some context on [my blog](http://olavnymoen.com/2016/10/30/attention-over-attention)

Reading comprehension for cloze style tasks is to remove word from an article summary, then read the article and try to infer the missing word. This example works on the CNN news dataset.

With the same hyperparameters as reported in the paper, this implementation got an accuracy of 74.3% on both the validation and test set, compared with 73.1% and 74.4% reported by the author.

To train a new model: `python model.py --training=True --name=my_model`

To test accuracy: `python model.py --training=False --name=my_model --epochs=1 --dropout_keep_prob=1`

Interesting parts
- Masked softmax implementation
- Example of batched sparse tensors with correct mask handling
- Example of pointer style attention
- Test/validation split part of the tf-graph
