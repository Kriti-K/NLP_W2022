
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

## [Overview](https://colab.research.google.com/drive/1pqRSOrfyt7uxev3SLYFKioloI7ElW4g4?usp=sharing)
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

You can also customize and add new entities according to your own dataset. For example, if you wanted to add entities for a narrowed down problem (like a dataset that has sentences targeting only one particular race or religion), you can do so using the ```train_entities.py``` file: 
```bash
python3 train_entities.py "filepath"
```
## Evaluation 
Since our system consists of two different models stacked one after the other, the evaluation methodology cannot be straightforward.  To evaluate our system, we created an aggregated dataset that contained the following types of texts:  
A: Texts that are both political and racially/religiously biased 
B: Texts that are only political but not racially/religiously biased
C: Texts that are not political but are racially/religiously biased
D:Texts that are neither political nor racially/religiously biased 
Then, the goal of our model becomes to identify A among all other categories. For example, we give a random news "a" to political ideology detection model, if the first model says "a" is political, then it gets passed to the entity sentiment analysis model. If the second model says "a" is racial, then we infer that "a" is from category A.
We already know "a" belongs to what category. If A is the correct category, this counts a true classification. Else it is a false prediction. 

And this way, our problem becomes a classification problem. 

### Evaluation Datasets

For A, we used the Social Bias Frames Dataset, which includes 642 texts that have both political and racial/religious bias. For selecting the B, C and D, we chose to sample inputs from the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset). This dataset includes 200k news articles labelled in multiple categories 

To compare our models with the state of the art, we used the [Ideological Books Corpus](https://people.cs.umass.edu/~miyyer/ibc/index.html) dataset to measure the performance for the first model that detects the political ideology. For the second model, we compared our method on 2 datasets, [Hate speech in US 2020 elections](https://arxiv.org/abs/2103.01664) and [ConvAbuse](https://arxiv.org/abs/2109.09483) dataset. 

### Baselines
Since there is no research work done that does exactly the same task as ours, there is no direct baseline that we can compare our models to. Thus, we need to compare both our models individually, even though we evaluate the entire system as a whole. 
For political ideology detection, we selected the [Recurrent Neural Network with Word2Vec](https://people.cs.umass.edu/~miyyer/ibc/index.html) embeddings proposed by M. Iyyer, 2014. 
For the Entity Sentiment Analysis, since the final output is a simple sentiment analysis problem, we compare our method with Google's pretrained [BERT](https://aclanthology.org/N19-1423/) model on the datasets mentioned above. 
### Results 
Below are the results when the entire system was evaluated according to our methodology. We use the **F1 Score, Precision and Recall** as our metrics of choice here. Even though the AUC is most preferred metric for classification tasks, it could not be generated here as we are combining output of 2 models here. Because of that, we are not able to generate absolute probabilities for the final output. 
| Metric| Score Value | 
| :---        |        ---: |          
| Precision      | 0.71| 
| Recall | 0.90| 
| F1 Score| 0.79| 
As seen in the table above, our proposed method has an F1 score of 0.79, meaning that our algorithm achieves some degree of success in identifying the political leaning of the articles and whether it uses a race or religion to manipulate its readers.

![alt text](https://raw.githubusercontent.com/Kriti-K/NLP_W2022/main/Evaluation/Result%20Photos/Picture1.png)

As seen in the image above, the model for Phase 1 (political ideology detection) that we propose (BERT without the repeating n-grams) performs the best out of the tested models. It provides a substantial 7% increase when compared to the baseline model (RNN+W2V).
![alt text](https://raw.githubusercontent.com/Kriti-K/NLP_W2022/main/Evaluation/Result%20Photos/Picture2.png)

The image above shows how our sentiment analysis pipeline stacks up against the state-of-the-art model (BERT) with respect to the F1 Score. We observe that our model does not out-perform the BERT model. One reason behind this is that we are limiting our sentiment analysis on only those entities that are considered political, racial or religious, but it might occur that there are no such entities but the text is still speaking positively or negatively about something. 
This is also translated on in the Second dataset (Hate Speech in 2020 elections) in the chart. Because we are using a dataset that has political, racial and religious entities, the F1 score is higher as compared to the ConvAbuse dataset. 

## Conclusion
From the above results, we can conclude that the system that we propose is able to successfully classify the political leaning of news/digital media articles and further detect whether the articles makes use of a particular race/religion to manipulate its voter-base. 
While the models do not achieve near-perfect metric scores, our system acts a stepping stone in this domain. This research can further be taken forward with more data analysis on the political ideology detection part and trying out different methodologies for entity sentiment analysis system.

