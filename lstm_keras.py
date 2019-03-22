import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

EMBEDDING_FILE = '/home/dante0shy/PycharmProjects/ToxicComments/embedding/glove.840B.300d.txt'
train_x = pd.read_csv('/home/dante0shy/PycharmProjects/ToxicComments/data/train.csv', index_col='id').fillna(" ")
test_x = pd.read_csv('/home/dante0shy/PycharmProjects/ToxicComments/data/test.csv', index_col='id').fillna(" ")

max_features = 100000
maxlen = 150
embed_size = 300

train_x['comment_text'].fillna(' ')
test_x['comment_text'].fillna(' ')
train_y = train_x[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
train_x = train_x['comment_text'].str.lower()

test_x = test_x['comment_text'].str.lower()

# Vectorize text + Prepare GloVe Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build Model
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = SpatialDropout1D(0.35)(x)

x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
# x = LSTM(256, return_sequences=True, dropout=0.15, recurrent_dropout=0.15)(x)
# x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Conv1D(256, kernel_size=3, padding='same', kernel_initializer='glorot_uniform')(x)
x = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='glorot_uniform')(x)
x = Conv1D(64, kernel_size=3, padding='same', kernel_initializer='glorot_uniform')(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])

out = Dense(6, activation='sigmoid')(x)

model = Model(inp, out)

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

early_stop = EarlyStopping(monitor = "accuracy", mode = "min", patience = 5)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prediction
batch_size = 64
epochs = 2

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks = [ early_stop])
predictions = model.predict(test_x, batch_size=batch_size, verbose=1)

submission = pd.read_csv('/home/dante0shy/PycharmProjects/ToxicComments/data/sample_submission.csv')
submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = predictions
submission.to_csv('submission.csv', index=False)