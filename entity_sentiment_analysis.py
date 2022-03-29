import pickle
from google.cloud import language_v1
import os
import numpy as np
from Levenshtein import distance as lsd
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/content/NLP_W2022/Code/gcp_creds.json"
client = language_v1.LanguageServiceClient()
language = "en"
encoding_type = "UTF8"

with open(r'/content/NLP_W2022/Entities/StereosetEntities.pkl','rb') as file_handle:
    ss_ents = pickle.load(file_handle)
    
def filter(entities):
    return_ents = []
    for i, entity in enumerate(entities):
        distances = [lsd(entity,word) for word in ss_ents]
        if min(distances) < 2: 
            return_ents.append(entity)
    return return_ents 

def get_entity_sentiment(text):
    ents = []
    sents = []
    document = {"content": text, "type": "PLAIN_TEXT", "language": language}
    response = client.analyze_entity_sentiment(document,encoding_type)
    for entity in response.entities:
            sentiment = entity.sentiment
            ents.append(entity.name)
            sents.append(sentiment.score)
    fents = filter(ents)
    if not fents:
        print('It does not speak about race or religion')
        exit()
    avg_sent = sum(sents)/len(fents)
    if avg_sent < 0:
        print('It speaks negatively about {}'.format(fents))
    elif avg_sent > 0:
        print('It speaks positively about {}'.format(fents))
    else:
        print('It is mostly neutral about {}'.format(fents))
        
        
