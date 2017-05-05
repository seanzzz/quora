#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re
import num2words
import json
import numpy as np
import pandas as pd

from random import random

from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


w2v = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)

train_1 = []
train_2 = []
is_dup_train = []

test_1 = []
test_2 = []
test_ids = []

MAX_LEN = 35
EMBEDDING_DIM = 300

with open('train.csv', 'rb') as csvfile:
    f = csv.reader(csvfile)
    next(f)
    for row in f:
        train_1.append(row[3])
        train_2.append(row[4])
        is_dup_train.append(int(row[5]))

with open('test.csv', 'rb') as csvfile:
    f = csv.reader(csvfile)
    next(f)
    for row in f:
        test_1.append(row[1])
        test_2.append(row[2])
        test_ids.append(row[0])

with open('gb_to_us.json', 'r') as f:
    try:
        gb_to_us = json.load(f)
    except ValueError:
        gb_to_us = {}


def to_US_spelling(word, dic, gb_to_us):
    if word in dic:
        return word
    if word.lower() in dic:
        return word.lower()
    elif word.lower() in gb_to_us:
        w = gb_to_us[word.lower()]
        return to_US_spelling(w, dic, gb_to_us)
        # otherwise do nothing
    if word.title() in dic:
        return word.title()
    if word.upper() in dic:
        return word.upper()
    if len(word) > 0 and word[-1] == 's':
        if word[:-1] in dic:
            return word[:-1]
        else:
            return to_US_spelling(word[:-1], dic, gb_to_us)
    else:
        return word


stop_words = ["a", "and", "of", "", " ", None, "to"]
all_sentences = []

removeRe = re.compile(ur"[!?.,'\"()/|-‘’;`<>…−“”？″′?]")


def process_sentence(s):
    """ given a sentence, remove unnecessary punctuations """
    """ then break the sentence into words in the dictionary """
    s = unicode(s, 'utf-8')
    s = removeRe.sub(" ", s, re.UNICODE)
    # s = re.sub(ur"\u2019", " ", s, re.UNICODE)
    s = re.sub(r"”", " ", s, re.UNICODE)
    s = re.sub(r"-", " ", s)
    s = re.sub(r":", " ", s)
    s = re.sub(r"{", " ", s)
    s = re.sub(r"}", " ", s)
    s = re.sub(r"\[", " ", s)
    s = re.sub(r"]", " ", s)
    s = re.sub(r"\$", " $ ", s)
    s = re.sub(r"\\", " $ ", s)
    s = re.sub(r"\%", " % ", s)
    s = re.sub(r"\&", " & ", s)
    s = re.sub(r"\+", " + ", s)
    s = re.sub(r"\^", " ^ ", s)
    s = re.sub(r"\=", " = ", s)
    s = re.sub(r"\*", " * ", s)
    s = re.sub(r"\_", " _ ", s)
    s = re.sub(r"\#", " # ", s)
    qs = re.split(r' |(\d+)', s)
    one_sentence = []
    for w in qs:
        if w in stop_words:
            continue
        if w in w2v:
            one_sentence.append(w)
            # one_sentence.append(w2v[w])
        else:
            # w_list = []
            try:  # convert to number
                w_int = int(w)
                w_word = num2words.num2words(w_int)
                w_list = re.split(r" |-|,", w_word)
                for word in w_list:
                    if word in w2v:
                        one_sentence.append(word)
                        # one_sentence.append(w2v[word])
            except ValueError:
                w = to_US_spelling(w, w2v, gb_to_us)
                if w in w2v:
                    one_sentence.append(w)
                    # one_sentence.append(w2v[w])
                else:
                    w_list = list(w)
                    for word in w_list:
                            # if word in w2v:
                            one_sentence.append(word)
                                # one_sentence.append(w2v[word])
                            # else:  # TODO: map unknown words with randomly initialized words
                            #     if word not in stop_words:
                            #         print word
                # else:
                #     if word not in stop_words:
                #         print word
    # if len(one_sentence) == 0:
    #     one_sentence.append([0] * 300)
    # one_sentence = np.array(one_sentence)
    # one_sentence = one_sentence.transpose()
    # # one_sentence = pad_sequences(one_sentence, MAX_LEN, dtype="float32")
    # one_sentence = one_sentence.transpose()
    if len(one_sentence) > MAX_LEN:  # remove duplicate if too long
        new_s = []
        [new_s.append(item) for item in one_sentence if item not in new_s]
        one_sentence = new_s
    return " ".join(one_sentence)


# train_1 = []



train_1 = [process_sentence(s) for s in train_1]
# train_2 = []
# test_1 = []
# test_2 = []
train_2 = [process_sentence(s) for s in train_2]
test_1 = [process_sentence(s) for s in test_1]
test_2 = [process_sentence(s) for s in test_2]

tokenizer = Tokenizer(filters="", lower=False)
tokenizer.fit_on_texts(train_1 + train_2 + test_1 + test_2)

sequences_1 = tokenizer.texts_to_sequences(train_1)
sequences_2 = tokenizer.texts_to_sequences(train_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_2)

data_1 = pad_sequences(sequences_1, maxlen=MAX_LEN)
data_2 = pad_sequences(sequences_2, maxlen=MAX_LEN)


# create embedding
nb_words = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    if word in w2v:
        embedding_matrix[i] = w2v[word]
    else:  # otherwise we need to come up with a random matrix
        embedding_matrix[i] = [random() for _ in range(300)]

num_lstm = 256
num_dense = 128
rate_drop_lstm = 0.2
rate_drop_dense = 0.2
act = 'relu'

# can also add first sequence to second, and add second to first to increase symmetry
STAMP = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm,
                                  rate_drop_dense)


embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LEN,
                            trainable=True)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm,
                  recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_LEN,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = LSTM(num_lstm, dropout=rate_drop_lstm,
          recurrent_dropout=rate_drop_lstm)(embedded_sequences_1)
x1 = LSTM(num_lstm, dropout=rate_drop_lstm,
          recurrent_dropout=rate_drop_lstm)(x1)

sequence_2_input = Input(shape=(MAX_LEN,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = LSTM(num_lstm, dropout=rate_drop_lstm,
          recurrent_dropout=rate_drop_lstm)(embedded_sequences_2)
y1 = LSTM(num_lstm, dropout=rate_drop_lstm,
          recurrent_dropout=rate_drop_lstm)(y1)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
# merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
# merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)

preds = Dense(1, activation='sigmoid')(merged)


########################################
# train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input],
              outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])

# model.summary()
print(STAMP)

# early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True,
                                   save_weights_only=True)

hist = model.fit([data_1, data_2], is_dup_train,
                 epochs=200, batch_size=1024, shuffle=True,
                 callbacks=[model_checkpoint])

# model.load_weights(bst_model_path)
# bst_val_score = min(hist.history['val_loss'])


# dic = {}
# for s in train_1:
#     if len(s) not in dic:
#             dic[len(s)] = 0
#     dic[len(s)] += 1

# sum_of_longer = 0
# for key in dic:
#     if key > MAX_LEN:
#         sum_of_longer += dic[key]

# print sum_of_longer


test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_LEN)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_LEN)

preds = model.predict([test_data_1, test_data_2], batch_size=2048, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=2048, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
submission.to_csv(STAMP + '.csv', index=False, columns=['test_id', 'is_duplicate'])
