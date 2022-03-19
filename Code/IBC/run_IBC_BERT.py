import tensorflow as tf
import tensorflow_text
from nltk.corpus import stopwords
import numpy as np
stop = stopwords.words('english')

model = tf.keras.models.load_model('C:\Users\Mrulay\OneDrive - University of Windsor\uWindsor\COMP 8730 - NLP\Research Paper\Datasets\PoliticalBias\full_ibc\IBC_BERT')

def preProcessText(text):
    text = text.lower()
    text = text.replace(r'[^\w\s]+', '') 
    text = text.replace('@', '')
    text = ' '.join(word.lower() for word in text.split() if word not in stop)
    return text

def pred(t):
    tx = preProcessText(t)
    out = np.argmax(model.predict([tx]))
    if out==1:
        return 'Neutral'
    elif out==2: 
        return 'Liberal'
    elif out==0:
        return 'Conservative'