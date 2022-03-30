import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from run_IBC_BERT import preProcessText
from sklearn.model_selection import train_test_split
import sys

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

learning_rate = 0.0001
optimizer = optimizers.Adam(learning_rate)
es = EarlyStopping(
    monitor='accuracy',
    patience=2,
    min_delta = 0.1,
    mode='max'
)

def train_bert(sents, labels):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # Neural network layers
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(3, activation='sigmoid', name="output")(l)

    # Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs = [l])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)
    model.fit(sents, labels, epochs=20, verbose=1, shuffle=True, callbacks=[es])
    return model

def eval_bert(model, X_test, y_test):
    print (model.evaluate(X_test, y_test))

if __name__ == '__main__':
    datapath = sys.argv[1]
    df = pd.read_csv(datapath)
    cols = df.columns
    sents = df[cols[0]]
    labels = df[cols[1]]
    
    sents = sents.apply(preProcessText)
    labels = labels = pd.get_dummies(labels)
    X_train, X_test, y_train, y_test = train_test_split(sents,labels)
    bertmodel = train_bert(X_train, y_train)
    eval_bert(bertmodel, X_test, y_test)