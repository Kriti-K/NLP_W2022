from run_IBC_BERT import preProcessText,pred
from entity_sentiment_analysis import get_entity_sentiment
import sys

if __name__ == '__main__':
    text = sys.argv[1]
    political_standing = pred(text)
    if political_standing == "Neutral":
        print("This document is politically neutral")
        exit()
    else:
        if political_standing == "Liberal":
            print("This document is biased towards Liberals.")
        elif political_standing == "Conservative":
            print("This document is biased towards Conservatives.")
    get_entity_sentiment(text)
        
    
