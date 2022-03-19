import pickle
from google.cloud import language_v1
import os
import numpy as np
from Levenshtein import distance as lsd
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/gcp_creds.json"
client = language_v1.LanguageServiceClient()
type_ = language_v1.types.Document.Type.PLAIN_TEXT
language = "en"
encoding_type = language_v1.EncodingType.UTF8

with open(r'C:\Users\Mrulay\OneDrive - University of Windsor\uWindsor\COMP 8730 - NLP\Research Paper\Datasets\RaceReligion\Stereoset\StereosetEntities.pkl','rb') as file_handle:
    ss_ents = pickle.load(file_handle)
    
def filter(entities):
    return_ents = []
    for i, entity in enumerate(entities):
        distances = [lsd(entity,word) for word in ss_ents]
        if min(distances) < 4: 
            return_ents.append(entity)
    return return_ents 

def get_entity_sentiment(text):
    ents = []
    sents = []
    document = {"content": text, "type_": type_, "language": language}
    response = client.analyze_entity_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    for entity in response.entities:
            sentiment = entity.sentiment
            ents.append(entity.name)
            sents.append(sentiment.score)
    fents = filter(ents)
    if not fents:
        print ('The document does not speak about race or religion')
        exit()
    avg_sent = sum(sents)/len(fents)
    if avg_sent < 0:
        print('The document speaks negatively about {}'.format(fents))
    if avg_sent > 0:
        print('The document speaks positively about {}'.format(fents))