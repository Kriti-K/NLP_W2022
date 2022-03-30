import pickle
from tracemalloc import reset_peak
from run_IBC_BERT import preProcessText
import spacy
import sys
import pandas as pd
import en_core_web_sm

if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    data_path = sys.argv[1]
    data = pd.read_csv(data_path)
    col = data.columns[0]
    sents = data[col].apply(preProcessText)
    ents = []
    for sent in sents:
        doc = nlp(sent) 
        for X in doc.ents:
            if X.label_=='NORP' or X.label_=='PERSON' or X.label_=='GPE' or X.label_=='LANGUAGE':
                ents.append(X.text)
    ents = list(set(ents))
    with open(r'/content/NLP_W2022/Entities/StereosetEntities.pkl', 'rb') as f1:
        x = pickle.load(f1)
    res = ents + x
    res = list(set(res))
    with open(r'/content/NLP_W2022/Entities/StereosetEntities.pkl', 'wb') as f1:
        pickle.dump(res, f1)