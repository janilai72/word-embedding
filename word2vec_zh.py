# -*- encoding: utf-8 -*-
'''
@File    :   word2vec_zh.py
@Time    :   2018/09/08 23:56:40
@Author  :   Jian.Lai
@Version :   1.0
@Contact :   jani.lai@aliyun.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''
import collections
import math
import random
import jieba
import os
import sys
import numpy as np
from six.moves import xrange
import tensorflow as tf


def read_data():
    raw_word_list = []
    with open('./data/shaoniangexing_seg.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        raw_word_list = line.split(' ')

    return raw_word_list

words = read_data()
print('Data size', len(words))

vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    print("word count:", len(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)

del words

print("Most common words(+UNK)", count[:5])
print("Sample data:", data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(batch_size, num_skips, skip_windows):
    """[summary]
    
    Args:
        batch_size ([type]): [description]
        num_skips ([type]): [description]
        skip_windows ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_windows
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_windows + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_windows
        targets_to_avoid = [skip_windows]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_windows]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_windows=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '-->',
          labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
skip_windows = 2
num_skips = 4
valid_size = 9
valid_window = 100
num_sampled = 64

valid_word = ['萧瑟', '天启城', '李寒衣', '明德帝', '五大监']
valid_examples = [dictionary[li] for li in valid_word]

graph = tf.Graph()

with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('gpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0/math.sqrt(embedding_size))
        )

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)

    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        inputs=embed,
        labels=train_labels,
        num_sampled=num_sampled,
        num_classes=vocabulary_size
    ))

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 3000000

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized...")

    average_loss = 0

    for step in xrange(num_steps):
        batch_inputs, bathc_labels = generate_batch(
            batch_size, num_skips, skip_windows)
        feed_dict = {train_inputs: batch_inputs, train_labels: bathc_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[:top_k]
                log_str = 'Nearest to %s:' % valid_word

                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


def plot_with_labels(low_dim_embs, labels, filename='images/tsne3.png', fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     fontproperties=fonts,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename, dpi=800)


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, fonts=font)
except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")