from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import (Flask, jsonify, send_file, send_from_directory, render_template, request)
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
from numpy import genfromtxt

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

def extract_labels_data(path):
    positive_reviews, positive_labels = get_reviews(path + "/CP/", positive=True)
    negative_reviews, negative_labels = get_reviews(path + "/CR/", positive=False)

    data = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels

    return labels, data

app = Flask(__name__)

@app.route("/sentiment_training", methods=['GET'])
# http://172.25.116.181:5000/sentiment_training?model_trainingpath=G:\Chen\sentiment\PortClass\&model_savepath=G:\Chen\sentiment\model.ckpt
def sentiment_training():
    print("Begin model training...")
    model_trainingpath = request.args.get('model_trainingpath')
    model_savepath = request.args.get('model_savepath')
    print("Model trainingpath: ", model_trainingpath)
    print("Model save path: ", model_savepath)
    
    URL_PATH = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    labels, data = extract_labels_data(model_trainingpath)
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

    num_epochs = 700
    batch_size = 60
    embedding_size = 100
    max_label = 2
    lstmUnits = 128

    embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embedding_matrix')
    embeddings = tf.nn.embedding_lookup(embedding_matrix, x)

    lstmCell = tf.nn.rnn_cell.LSTMCell(lstmUnits,state_is_tuple=True)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)

    value, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, max_label]), name='weight')
    bias = tf.Variable(tf.constant(0.1, shape=[max_label]), name='bias')
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
    
    # Save model ****
    saver = tf.train.Saver({'embedding_matrix': embedding_matrix, 'weight':weight, 'bias':bias}, save_relative_paths=True)

    with tf.Session() as session:
        session.run(init)
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
            if (epoch + 1) > 50 and test_acc > 0.75 and test_loss < 0.7:
                print("Model ends here.")
                save_path = saver.save(session, model_savepath)
                print("Save to path: ", save_path)
                break
    return "Success!"
        
@app.route("/sentiment_model", methods=['POST'])
# 172.25.116.181:5000/sentiment_model
# {
#     "model_path" : "G:\\Chen\\sentiment\\model.ckpt",
#     "model_testpath" : "G:\\Chen\\sentiment\\results\\0_test_data.csv",
#     "result_path" : "G:\\Chen\\sentiment\\results\\0_test_prediction.csv"
# }
def sentiment_model():
    paths = request.get_json()
    if "model_path" in paths and 'model_testpath' in paths and 'result_path' in paths:
        model_testpath = paths['model_testpath']
        model_path = paths['model_path']
        result_path = paths['result_path']
        print('model_path: ' + paths['model_path'])
        print('model_testpath: ' + paths['model_testpath'])
        print('result_path: ' + paths['result_path'])

        tf.reset_default_graph()
        
        MAX_SEQUENCE_LENGTH = 717
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)
        vocabulary_size = len(vocab_processor.vocabulary_)
        embedding_size = 100
        max_label = 2
        lstmUnits = 128
    
        test_x = genfromtxt(model_testpath, delimiter=',').reshape(-1, 1)
        test_x = tf.convert_to_tensor(test_x, np.int32)
            
        embedding_matrix = tf.get_variable('embedding_matrix', shape=[994, 100])
        embeddings = tf.nn.embedding_lookup(embedding_matrix, test_x)
        
        lstmCell = tf.nn.rnn_cell.LSTMCell(lstmUnits,state_is_tuple=True)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)
 
        value, _ = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)
    
        weight = tf.get_variable('weight', shape=[lstmUnits, max_label])
        bias = tf.get_variable('bias', shape=[max_label])
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        
        saver = tf.train.Saver()
        print("Begin restore")
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            print("Restored")
            
            print('bias: ', bias.eval())
            print('weight: ', weight.eval())
            print('last: ', last.eval())
            print(tf.matmul(last, weight).eval(session=sess))
#             y_ = (tf.matmul(last, weight).eval(session=sess) + b.eval(session=sess))
#             print(y_)
            
#             print("Need to write prediction to result path...") 
#             np.savetxt(result_path, y_, delimiter=",")

            return "Success."

if __name__ == '__main__':
    app.run(host='0.0.0.0')