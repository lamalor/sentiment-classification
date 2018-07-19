from __future__ import absolute_import, division, print_function
from flask import (Flask, jsonify, send_file, send_from_directory, render_template, request)
from sklearn.model_selection import train_test_split
from six.moves import urllib
import os
import csv
import numpy as np
from numpy import array
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, Dense, Dropout, LSTM, MaxPooling1D, Convolution1D
from keras.models import model_from_json
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

proxy = 'gw-proxy-la03p.corp.tcw.com:80'
os.environ['https_proxy'] = proxy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_reviews(dirname, label):
    reviews = []
    labels = []
    file_name = []
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            with open(dirname + filename, 'r+', encoding='utf-8-sig') as f:
                review = f.read().lower()
                reviews.append(review)
                labels.append(label)
                file_name.append(filename)
    return reviews, labels, file_name

def extract_labels_data(styles):
    data = []
    labels = []
    file_names = []
    for i, style in enumerate(styles):
        review, label, file_name = get_reviews("G:/Chen/PortClass/"+style+"/",label=i)
        data += review
        labels += label
        file_names += file_name
        style_dict[i] = style
    return labels, data, file_names

style_dict = {}
style_list = ['CP','CPOP','CPPE','CPPL','CR','CRRV']
num_styles = len(style_list) 
labels, data, file_name = extract_labels_data(style_list)
labels = to_categorical(array(labels))

app = Flask(__name__)

@app.route("/sentiment_training", methods=['GET'])
# http://172.25.116.181:5000/sentiment_training?model_trainingpath=0&model_savepath=G:/Chen/sentiment/model
def sentiment_training():
    print("Begin model training...")
    model_trainingpath = int(float(request.args.get('model_trainingpath')))
    model_savepath = request.args.get('model_savepath')

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(data)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(data)
    #print(encoded_docs)
    max_length = max([len(i.split(' ')) for i in data])
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(padded_docs, labels, range(0, len(labels)), test_size=0.2, random_state=model_trainingpath)

    # define the model
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = 90, input_length = max_length))

    # Added LSTM Below 
    model.add(Conv1D(filters=32, kernel_size=4, activation='elu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=32, kernel_size=2, activation='elu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(LSTM(120, return_sequences=True))
    model.add(LSTM(60, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(64, kernel_initializer='normal', activation='elu'))
    model.add(Dense(num_styles, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    model_json = model.to_json()
    
    print("Saving the model...")
    with open(model_savepath + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_savepath + ".h5")
    print("Saved model to disk")
    
    return "Success!"
        
@app.route("/sentiment__model", methods=['POST'])
# http://172.25.116.181:5000/sentiment__model
# {
#     "model_savepath" : "G:\\Chen\\sentiment\\model",
#     "model_testpath" : "0",
#     "results_path" : "G:\\Chen\\sentiment\\results\\0"
# }
def sentiment_model():
    K.clear_session()
    paths = request.get_json()
    if "model_savepath" in paths and 'model_testpath' in paths and 'results_path' in paths:
        model_savepath = paths['model_savepath']
        model_testpath = int(float(paths['model_testpath']))
        results_path = paths['results_path']
        print('model_savepath: ' + paths['model_savepath'])
        print('model_testpath: ' + paths['model_testpath'])
        print('results_path: ' + paths['results_path'])
        
        # prepare tokenizer
        t = Tokenizer()
        t.fit_on_texts(data)
        vocab_size = len(t.word_index) + 1
        encoded_docs = t.texts_to_sequences(data)
        max_length = max([len(i.split(' ')) for i in data])
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        json_file = open(model_savepath+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_savepath+".h5")
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Loaded model.")

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(padded_docs, labels, range(0, len(labels)),test_size=0.2, random_state=model_testpath)
        indice_mappings = np.asarray(file_name)[indices_test] 
        style_mappings = [style_dict[np.argmax(list(map(lambda x: (round(x)), i)))] for i in loaded_model.predict(X_test)]
        style_mappings_actual = [style_dict[np.argmax(list(map(lambda x: (round(x)), i)))] for i in y_test]
        
        # pred = prediction ['portfolio.txt', 'portfolio style']
        # actual = actual ['portfolio.txt', 'portfolio style']
        pred = list()
        actual = list()
        for i, x in enumerate(indice_mappings):
            pred.append([indice_mappings[i], style_mappings[i]])
            actual.append([indice_mappings[i], style_mappings_actual[i]])
            
        # Save prediction results and actual results.
        # ['portfolio.txt', 'portfolio style'], ['...'] 
        with open(results_path + "_pred.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(pred)
        with open(results_path + "_test.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(actual)
        print("Model saved at: ", results_path + "_(pred and test).csv")

        print("Prediction: ", pred)
        print("Actual: ", actual)
        score, acc = loaded_model.evaluate(X_test, y_test,batch_size=20, verbose=0)
        print("Accuracy: ", acc * 100)
        return jsonify(pred)
    else: 
        return "Missing one."

if __name__ == '__main__':
    app.run(host='0.0.0.0')
