from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import os
import random
import tarfile
import re
from six.moves import urllib
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf

proxy = 'gw-proxy-la03p.corp.tcw.com:80'
os.environ['https_proxy'] = proxy

DOWNLOADED_FILENAME = 'ImdbReviews.tar.gz'

def download_file(url_path):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path, DOWNLOADED_FILENAME)

    print('Found and verified file from this path: ', url_path)
    print('Downloaded file: ', DOWNLOADED_FILENAME)

TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")

def get_reviews(dirname, positive=True):
    label = 1 if positive else 0

    reviews = []
    labels = []
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            with open(dirname + filename, 'r+', encoding='utf-8') as f:
                review = f.read().lower()
                reviews.append(review)
                labels.append(label)
    return reviews, labels           

def extract_labels_data():
    positive_reviews, positive_labels = get_reviews("G:/Mohit/PortClass/CP/", positive=True)
    negative_reviews, negative_labels = get_reviews("G:/Mohit/PortClass/CR/", positive=False)

    data = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels

    return labels, data

URL_PATH = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
labels, data = extract_labels_data()
max_document_length = max([len(x.split(" ")) for x in data])

MAX_SEQUENCE_LENGTH = 717

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)

x_data = np.array(list(vocab_processor.fit_transform(data)))
y_output = np.array(labels)

vocabulary_size = len(vocab_processor.vocabulary_)
vocab_dict = vocab_processor.vocabulary_._mapping

np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]

TRAIN_DATA = 60
TOTAL_DATA = 69

train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]

test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]

tf.reset_default_graph()

x = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int32, [None])

num_epochs = 10000
batch_size = 60
#1: 10, #2: 60 -> 0.555567
embedding_size = 100
max_label = 2
lstmUnits = 128

embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embeddings = tf.nn.embedding_lookup(embedding_matrix, x)

lstmCell = tf.nn.rnn_cell.LSTMCell(lstmUnits,state_is_tuple=True)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)

value, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, max_label]))
bias = tf.Variable(tf.constant(0.1, shape=[max_label]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
logits = (tf.matmul(last, weight) + bias)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(cross_entropy)

prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(0.015)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    init.run()
    for epoch in range(num_epochs):
        
        num_batches = int(len(train_data) // batch_size) + 1
        
        for i in range(num_batches):
            # Select train data
            min_ix = i * batch_size
            max_ix = np.min([len(train_data), ((i+1) * batch_size)])

            x_train_batch = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]
            
            train_dict = {x: x_train_batch, y: y_train_batch}
            session.run(train_step, feed_dict=train_dict)
            
            train_loss, train_acc, logits_val, emb_mat = session.run([loss, accuracy, logits, embedding_matrix], feed_dict=train_dict)

        test_dict = {x: test_data, y: test_target}
        test_loss, test_acc = session.run([loss, accuracy], feed_dict=test_dict)    
        print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.5}'.format(epoch + 1, test_loss, test_acc)) 
        if (epoch + 1) > 600 and test_acc > 0.75 and test_loss < 0.7:
            print("Model ends here.")
            # Store model and its weights
            break