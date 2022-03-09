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

The dataset has many pitfalls that will affect our modeling, and have dramatically affected the accuracy results to be overinflated and underpromising.

1. Sample vs. Population Distributions

2. Class Imbalance - 

3. Messy text data - the posts include metions about the MBTI types which can be a proxy for the actual MBIT target label

4. Not enough data - certain forums might be discussing certain topics which can be a proxy for the personality type as opposed to a causation. 

## Modeling

## Results

## Online Web App

I deployed an online application to showcase the model in action. 

The application will predict user personality as either *THINKING* or *FEEELING* based on user text input. 
The app will return a bar chart of personality probabilites and top word weights by TF-IDF scores. 

The application will also 50 posts from a user to run throught the model with the same output results. 

![App-Screenshot](https://share.streamlit.io/westonshuken/personality-prediction/main/mbti_app.py)

*The purpose of this is to inspect which words might be used to make predictions.*

You can try the app [HERE](https://share.streamlit.io/westonshuken/personality-prediction/main/mbti_app.py).

## Applications of the Predictive Model

There are numerous applications for using this personality predictive model:

- Customer Segmentation
- Digital Advertising 




## Recommednations

## Next Steps

## Reproducability 

This repository uses Python version 3.8.5

The dataset can be found [Kaggle](https://www.kaggle.com/datasnaek/mbti-type) or in the `data` folder on the repository

Using environement.yml file will allow to build the environment in which works for this repository and all of the notebooks

The requirements.txt file is used specifically for the Heroku APP deployment

**Repository Structure:**
```
├── data preprocessing                     <- Team Member's individual notebooks 
├── data                                   <- Both sourced externally and generated from code 
├── images                                 <- Both sourced externally and generated from code 
├── .gitignore                             <- gitignore 
├── GridSearch.ipynb                       <- Supplementary documentation of gridsesarching optimal parameters
├── GridSearchSMOTE.ipynb                  <- Supplementary documentation of gridsesarching optimal parameters using SMOTE
├── README.md                              <- The top-level README for reviewers of this project
├── index.ipynb                            <- Narrative documentation of analysis in Jupyter notebook
└── presentation.pdf                       <- PDF version of project presentation
