import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import re
import pickle

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag


from textblob import TextBlob



from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from imblearn.under_sampling import RandomUnderSampler

SEED = 53188535
class preprocess:

    def url_remove(self, post):
        return re.sub(r'http\S+', '', post)

    def pipe_remove(self, post):
        return re.sub(r'[|]', ' ', post)

    def punc_remove(self, post):
        return re.sub(r'[\'_:]', '', post)

    def remove_dig_token(self, post):
        return [post[i] for i in range(len(post)) if post[i].isalpha()]

    def remove_stopwords(self, post, words=None):
        sw = stopwords.words('english')
        if words:
            sw.extend(words)
        return [post[i] for i in range(len(post)) if post[i] not in sw]

    def replace_mbti(self, post, regxx='(intp)|(intj)|(entp)|(entj)|(infj)|(infp)|(enfj)|(enfp)|(istj)|(isfj)|(estj)|(esfj)|(istp)|(isfp)|(estp)|(esfp)|(intp)'):
        new = re.sub("r" + regxx, "", post)
        return new

    def remove_symbols(self, post):
        encoded_string = post.encode('ascii', 'ignore')
        deconded_string = encoded_string.decode()
        return deconded_string

    def lemmatize_text(self, tokens): 
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w) for w in tokens]

    def join_tokens(self, tokens):
        long_string = ' '.join(tokens)
        return long_string
        
    def spelling(self, posts):
        b = TextBlob(posts)
        return str(b.correct())

    def lemmend_pos(self, tokens, pos=True):

        def get_wordnet_pos(treebank_tag):
            '''
            Translate nltk POS to wordnet tags
            '''
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            elif treebank_tag:
                return wordnet.NOUN
        if pos == True:
            lemmatizer = WordNetLemmatizer()
            tagged = pos_tag(tokens)
            tagged = [(token[0], get_wordnet_pos(token[1])) for token in tagged]
            lemmed = [lemmatizer.lemmatize(token[0], token[1]) for token in tagged]
        elif pos != True:
            lemmatizer = WordNetLemmatizer()
            lemmed = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmed

class run_models:
    '''
    '''
    def run(self, df, X_column, targets, models, table, tfidf=False, SEED=SEED):
        '''
        '''
        for target in targets:
            print('-'*20)

            X = df[X_column]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=df[target], random_state=SEED)

            print(f"{target}, target balance: \n{y_train.value_counts(normalize=True)}")

            if tfidf == True:
                print(f'Vectorizing....  @ {time.asctime()}')
                vc = TfidfVectorizer(ngram_range=(1,2))
                X_train_vc = vc.fit_transform(X_train)
                X_test_vc = vc.transform(X_test)
                
            else:
                print(f'Vectorizing....  @ {time.asctime()}')
                vc = CountVectorizer(ngram_range=(1,2))
                X_train_vc = vc.fit_transform(X_train)
                X_test_vc = vc.transform(X_test)

            i = 0
            for clf in models:  
            
                model_name = str(clf)
                print(f'Working on {model_name} for {target} @ {time.asctime()}')

                models[clf].fit(X_train_vc, y_train)
                cv_score = cross_val_score(models[clf], X_train_vc, y_train, cv=5)
                cv_score_mean = round(np.mean(cv_score), 4)
            
                y_pred = models[clf].predict(X_test_vc)
                acc_score = accuracy_score(y_pred, y_test)
                row = pd.DataFrame({'Model': model_name + "_" + target, 'Target': target, 'CVScore': round(cv_score_mean, 4), \
                                                    'TestAcc': round(acc_score, 4)}, index=[i])
                table = pd.concat([table, row], ignore_index=True)
                i = i + 1

        return table


    def run_usampled(self, df, X_column, targets, models, table, tfidf=False, SEED=SEED):
        '''
        '''
        
        for target in targets:
            train_set, test_set = train_test_split(df, random_state=SEED, stratify=df[target])

            X_train = train_set[X_column]
            X_train = np.array(X_train).reshape(-1, 1)

            y_train = train_set[target]
            y_train = np.array(y_train).reshape(-1, 1)

            X_test = test_set[X_column]
            X_test = np.array(X_test).reshape(-1, 1)

            y_test = test_set[target]
            y_test = np.array(y_test).reshape(-1, 1)

            # instantiating the random undersampler
            rus = RandomUnderSampler() 

            # resampling training set X & y
            X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
            X_test_rus, y_test_rus = rus.fit_resample(X_test, y_test)


            # new class distribution
            print(f"Train: {np.unique(y_train, return_counts=True)}")
            print(f"Train: {np.unique(y_train_rus, return_counts=True)}")
            print(f"Test: {np.unique(y_test, return_counts=True)}")
            print(f"Test: {np.unique(y_test_rus, return_counts=True)}")

            X_train_rus = pd.Series(X_train_rus.reshape(-1))
            y_train_rus = pd.Series(y_train_rus.reshape(-1))

            X_test_rus = pd.Series(X_test_rus.reshape(-1))
            y_test_rus = pd.Series(y_test_rus.reshape(-1))

            if tfidf == True:
                print(f'Vectorizing....  @ {time.asctime()}')
                vc = TfidfVectorizer(ngram_range=(1,2))
                X_train_vc = vc.fit_transform(X_train_rus)
                X_test_vc = vc.transform(X_test_rus)
                
            else:
                print(f'Vectorizing....  @ {time.asctime()}')
                vc = CountVectorizer(ngram_range=(1,2))
                X_train_vc = vc.fit_transform(X_train_rus)
                X_test_vc = vc.transform(X_test_rus)

            i = 0
            for clf in models:  
                
                model_name = str(clf)
                print(f'Working on {model_name} @ {time.asctime()}')

                models[clf].fit(X_train_vc, y_train_rus)

                cv_score = cross_val_score(models[clf], X_train_vc, y_train_rus, cv=5)
                cv_score_mean = round(np.mean(cv_score), 4)

                y_pred = models[clf].predict(X_test_vc)
                acc_score = accuracy_score(y_pred, y_train_rus)

                row = pd.DataFrame({'Model': model_name + "_" + target, 'Target': target, 'CVScore': round(cv_score_mean, 4), \
                                                    'TestAcc': round(acc_score, 4)}, index=[i])
                table = pd.concat([table, row], ignore_index=True)
                i = i + 1


        return table