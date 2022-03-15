import pandas as pd 
import json 


    
def jsonToList(content):

    data = content['data']['intrasentence']
    data1 = content['data']['intersentence']
    sentence = []
    label = []
    type = []
    for i in range(0, len(data)):
        if data[i]['bias_type'] == "race" or data[i]['bias_type'] == "religion":
            for j in range(0, len(data[i]['sentences'])):
                sentence.append(data[i]['sentences'][j]['sentence'])
                label.append(data[i]['sentences'][j]['gold_label'])
                type.append(data[i]['bias_type'])
                
    for i in range(0, len(data1)):
        if data1[i]['bias_type'] == "race" or data1[i]['bias_type'] == "religion":
            for j in range(0, len(data1[i]['sentences'])):
                to_sent = data1[i]['context'] +' '+ data1[i]['sentences'][j]['sentence']
                sentence.append(to_sent)
                label.append(data1[i]['sentences'][j]['gold_label'])
                type.append(data1[i]['bias_type'])
    return sentence, label, type 
                
def makeCsv(sentence, label, type):
    d = {'sentences':sentence, 'label':label, 'type':type}
    df = pd.DataFrame(d)
    df.to_csv('Stereoset.csv')

if __name__ == '__main__': 
    
    with open('StereoSet.json') as f: 
        content = json.load(f)
        
    sentence, label, type = jsonToList(content)
    makeCsv(sentence, label, type)