# Mitigating Stereotypical Bias and Political Polarization

This repository stores the source code and datasets for our research project. The aim for this project is to detect whether race and religion are used to manipulate the readers of a political article. 

## Overview
The system takes in a text input of an article and undergoes 2 steps of processing: 
1. First the political polarity of the article is determined.
2. Then, all references to racial and religious bias are extracted and each of their sentiments are calculated

For example:
```bash
Input: Forcing middle-class workers to bear a greater share of the cost of government weakens their support for needed investments and stirs resentment toward those who depend on public services the most .


Output: This document is biased towards Liberals.
It does not speak about any race or religion
```

## Python Dependencies

- [Tensorflow](https://www.tensorflow.org/)
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language)
- [Spacy](https://spacy.io/)
- [Nltk](https://www.nltk.org/)
- [Python-Levenshtein](https://pypi.org/project/python-Levenshtein)

## Installing and Preparing
1. Use the simple git clone command to close the repository. 
2. Download the files in [this](https://uwin365-my.sharepoint.com/:f:/g/personal/mistry13_uwindsor_ca/EsQc9-yZyRJHl3SQcMbh_sMBVkAE-ZZwyEQCvtIjZqQehw) folder and save them to ``Models/IBC_BERT`` folder


## Executing the code
 and run a simple bash script to run the code with desired input.
```bash
python3 run.py "<input text>"
```
