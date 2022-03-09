# Automatic Personality Prediction

#### by Weston Shuken


![opening-image]()

---

## Overview

Automatic personality detection is the automated forecasting of a personality using human-generated or exchanged contents: text, speech, videos, images, etc.

The purpose of this project is use automatic persnoality detection using text data from forum-based websites. Moreover, I will use word vectorization and machine learning models to disvoer if personality *can* be detected by a user's word usage and word choice frequencies. 

## Opportunities


## Data and Methods

The data used to make predictions is a collection of 50 posts per ~8,600 users on a forum-based website: [Personality Cafe](https://www.personalitycafe.com/). All of the users have self-tagged their profiles with thier Myers-Briggs Type Indicator (MBTI) personality. These personality types will be used as the **target** variables, and the collection of posts will be the **predictor** variables.

The Myers-Briggs test is a very popular test that ask users approximately 90 questions about their behavior and assigns the user a type of personality based on this assessment. This test takes around 20-30 for someone to complete. 

There are 16 different personality types using a combination of 8 overall traits. See below:

    Introversion (I) **vs** Extroversion (E)
    Intuition (N) **vs** Sensing (S)
    Thinking (T) **vs** Feeling (F)
    Judging (J) **vs** Perceiving (P)
    
*If you are unfamilar with the MBTI, please visit Myers-Briggs Type Indicator]() for more info*

Various methods were used to preprocess, vectorize, and predict:

Preprocessing Methods:

    Lowercase all words
    Remove Punctuation
    Remove non-ASCII characters
    
Vectorization Strategies:
    
    Bag of Words
    Term Frequency-Inverse Document Frequency (TF-IDF)
    Doc2Vec (from Gensim)
    
Machine Learnging Models:
    
    Stochastic Gradient Descent Classifier
    Logistic Regression
    Random Forest
    Mutlinominal Naive Bayes
    
 Evaluation Metrics:
    
    Accuracy
    F1-Score
    Testing on Reddit data


"Automatic personality prediction has become a widely discussed topic for researchers in the NLP community."

## Understanding the Dataset

## Modeling

## Results

## Online Web App

## Applications of the Predictive Model

There are numerous applications for using this personality predictive model:

- Customer Segmentation
- Digital Advertising 




## Recommednations

## Next Steps

## Reproducability 

This repository uses Python version 3.8.5

The dataset can be found [Kaggle]() or in the `data` folder on the repository

Using environement.yml file will allow to build the environment in which works for this repository and all of the notebooks

The requirements.txt file is used specifically for the Heroku APP deployment
