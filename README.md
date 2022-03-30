
# Mitigating Stereotypical Bias and Political Polarization

The internet plays an important role in influencing political beliefs in the voter base. The most prominent ways how the people are being manipulated is through news articles and social media. We propose a system that aims to classify whether an article or a social media post uses religion or race to manipulate political standing of it’s readers. The system follows a two-phased procedure; the first phase includes classification of sentences in either of the political groups ``Liberal, Neutral, or Conservative``, while the second phase focuses on detecting the specific race or religion being targeted, and whether those are used positively or negatively. This repository stores the source code and datasets for our research project.  

## Python Dependencies

- [Tensorflow](https://www.tensorflow.org/)
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language)
- [Spacy](https://spacy.io/)
- [Nltk](https://www.nltk.org/)
- [Python-Levenshtein](https://pypi.org/project/python-Levenshtein)

## [Setup](https://colab.research.google.com/drive/1EQ7-7QyUd1A7B6zIB5p6mK0ZIZYHd3IB?authuser=1#scrollTo=kr8pSED4YSPv)
To run our code, you need to have ```python >= 3.8``` installed. You can then use ```pip ``` to install all the required dependencies that are listed in ```requirements.txt```. 
Step 1: Clone this github repository and set it as your working directory by the following command:
```bash
git clone https://github.com/Kriti-K/NLP_W2022.git
%cd NLP_W2022
```
Step 2: Install all the dependencies from the ```requirements.txt```
```bash
pip install -r requirements.txt
```
Step 3: Download the required files using the ```wget``` commands
```bash
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pVViO4phYWIJ2UgC_xaZrU4Y_1fVcNDF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pVViO4phYWIJ2UgC_xaZrU4Y_1fVcNDF" -O /content/NLP_W2022/Models/IBC_BERT/variables/variables.data-00000-of-00001 && rm -rf /tmp/cookies.txt

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A4cSYIi5fak-dMmP5uNaYeGTAbbk-rg9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A4cSYIi5fak-dMmP5uNaYeGTAbbk-rg9" -O /content/NLP_W2022/Code/gcp_creds.json && rm -rf /tmp/cookies.txt

!python -m nltk.downloader all
``` 

```bash
python3 run.py "<input text>"        
```
A tutorial notebook is available [here](https://colab.research.google.com/drive/1EQ7-7QyUd1A7B6zIB5p6mK0ZIZYHd3IB?authuser=1#scrollTo=kr8pSED4YSPv) that displays the execution of all these steps and performs testing of the code as well. 

## Overview
The system takes in a text input of an article and undergoes 2 steps of processing: 
1. First the political polarity of the article is determined.
	For this step, the dataset of our choice was the [Ideological Books Corpus(IBC)](https://people.cs.umass.edu/~miyyer/ibc/index.html) with sub-sentential annotations[3]. It includes 4,062 sentences annotated for political ideology at a sub-sentential level. Specifically, it contains 2025 liberal sentences, 1701 conservative sentences, and 600 neutral sentences. 
	We cleaned and modelled our corpus on various machine learning and deep learning models before we settled with the BERT model[2]. BERT produces state-of-the-art results in many NLP tasks. BERT uses bi-directional training approach, hence, it has a better knowledge of language context and flow rather than other single directional models.
2. Then, all references to racial and religious bias are extracted and each of their sentiments are calculated:
	First, we synthesize a custom set of entities from two datasets; (i) [Social Bias Frames (SBIC)](https://arxiv.org/abs/1911.03891) and (ii) [Stereoset](https://arxiv.org/abs/2004.09456). Both these datasets contain a substantial amount of sentences that have either religious or racial bias in them.
	We use our very own dataset to perform entity level sentiment analysis to for each entity that has reference to any race or religion to get their exact sentiment. 

For example:
```bash
Input: Forcing middle-class workers to bear a greater share of the cost of government weakens their support for needed investments and stirs resentment toward those who depend on public services the most .


Output: This document is biased towards Conservatives. 
		It does not speak about race or religion
```

To see the detailed execution and explanation of every part of our code, refer to [this](https://colab.research.google.com/drive/1pqRSOrfyt7uxev3SLYFKioloI7ElW4g4?usp=sharing) notebook.

## Train and test your own dataset
To train and test your own dataset on our model architecture, the dataset needs to be in this specification:
 - It has to be in a ```.csv``` file with 2 columns, the first should contain the sentences and the second should contain their labels. 
 - There should be 3 categories in the dataset, ``Liberal, Neutral, and Conservative``. 
 Example dataset: 

| Sentences| Lables | 
| :---        |        ---: |          
| xyz      | liberal| 
| abc| conservative        | 
| pqr| neutral        | 

To train your own dataset, you can run the following command in the terminal:
```bash
python3 new_train_test.py "filepath"
```
