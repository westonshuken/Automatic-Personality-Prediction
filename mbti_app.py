###################### IMPORTS ################################
from unittest.util import strclass
import streamlit as st
import pickle
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

################################ LOADING AND FUNCTIONS ################################

load_model = pickle.load(open('./models/final_model.pkl', 'rb'))

def prediction(text):
    pred = str(load_model.predict([text]))
    return pred

def likelihood(text):
    likelihood = str(round(max(load_model.predict_proba([text])[0])*100, 2))
    return likelihood

################################ FORM COLLECT DATA ################################

with st.form("Form"):
    st.write("Would you mind answering a few questions?")
    text_input1 = st.text_area('How are you doing today?')
    text_input2 = st.text_area('What is your favorite thing about today?')
    text_input3 = st.text_area('What is your least favorite thing about today?')
    text_input4 = st.text_area('Anything else on your mind? Write anything you want :-)')

    submitted = st.form_submit_button("Submit")
    total_text = str(text_input1 + text_input2 + text_input3 + text_input4)
    if submitted and len(total_text) > 100:
        sentiment = prediction(total_text)
        lhood = likelihood(total_text)
        if sentiment == "['f']":
            st.markdown("##### The model believes it is a " + lhood + "% probabilty you have a *feeling* personality")
                #     st.image('images/happy_gogo.png', width=200)
        elif sentiment == "['t']":
            st.markdown('##### You have a *thinking* personalityl')
                    # st.image('images/neutral_gogo.png', width=200)
                # else:
                #     st.markdown('##### This tweet is negative')
                #     st.image('images/cry_gogo.png', width=200)
                # else:
                #     st.markdown("###### *don't be shy*")
        st.write('The model believes it is a ', lhood,"% change this is correct.")
    elif submitted and len(total_text) <= 100:
        st.write("Can you please write a bit more?")
        st.text(str(len(total_text)))


################################ USER PREDICTION ################################

text_input1 = st.text_area('How are you doing today?')
btn1 = st.button('Move on')
if btn1:
    text_input2 = st.text_area('What is your favorite thing about today?')
    btn2 = st.button('Next Question')
    if btn2:
        text_input3 = st.text_area('What is your least favorite thing about today?')
    btn3 = st.button('Last Question')
    if btn3:
        text_input4 = st.text_area('Anything else on your mind? Write anything you want :-)')
    show_sentiment = st.button('Blast Off')
    if show_sentiment:
        sentiment = prediction(text_input1)
        if sentiment == "['f']":
            st.markdown('##### You have a *feeling* personality')
                #     st.image('images/happy_gogo.png', width=200)
        elif sentiment == "['t']":
            st.markdown('##### You have a *thinking* personalityl')
                    # st.image('images/neutral_gogo.png', width=200)
                # else:
                #     st.markdown('##### This tweet is negative')
                #     st.image('images/cry_gogo.png', width=200)
                # else:
                #     st.markdown("###### *don't be shy*")

        lhood = likelihood(text_input1)
        st.write('  ', lhood,"% probabilty this is correct")


################################ USER ANSWER ################################

st.markdown('# Are you a *thinking* or *feeling* type of personality?')

st.write('')
st.write('')
st.write('')

col1, col2 = st.columns(2)
st.write('')
st.write('')
think = col1.button('Thinking')
feel = col2.button('Feeling')


if think:
    st.markdown("When you make a decision, you like to find the basic truth or principle to be applied, regardless of the specific situation involved. You like to analyze pros and cons, and then be consistent and logical in deciding. You try not to be impersonal, so you won't let my personal wishes--or other people's wishes--influence me.")
    
#     tru = st.checkbox("True")
#     ntru = st.checkbox("Not True")
# if tru is True:
#     st.write("Very cool, you really know yourself. This information about the `Feeling` personality comes direcetly from the Myers-Briggs Type Indecator website.")
# if ntru is True:
#     st.write("Fair enough. This comes direcetly from the Myers-Briggs Type Indecator website.")

if feel:
    st.markdown("You believe you can make the best decisions by weighing what people care about and the points-of-view of persons involved in a situation. You are concerned with values and what is the best for the people involved. You like to do whatever will establish or maintain harmony. In your relationships, you appear caring, warm, and tactful.")
#     tru = st.checkbox("True")
#     ntru = st.checkbox("Not True")

# if tru is True:
#     st.write("Very cool, you really know yourself. This information about the `Feeling` personality comes direcetly from the Myers-Briggs Type Indecator website.")
# if ntru is True:
#     st.write("Fair enough. This comes direcetly from the Myers-Briggs Type Indecator website.")

st.markdown("---")

###################### ORIGINAL USER PREDICTION ################################

# st.markdown("## Let's use our model to see if we can predict right")

# text_input = st.text_area('How are you feeling today?')

# show_sentiment = st.button('Send')

# load_model = pickle.load(open('./models/final_model.pkl', 'rb'))


# def prediction(text):
#     pred = str(load_model.predict([text]))
#     return pred

# def likelihood(text):
#     likelihood = str(round(max(load_model.predict_proba([text])[0])*100, 2))
#     return likelihood

# if show_sentiment:
#     sentiment = prediction(text_input)
#     # st.text(sentiment)
#     # st.text(type(sentiment))
#     if sentiment == "['f']":
#         st.markdown('##### You have a *feeling* personality')
#     #     st.image('images/happy_gogo.png', width=200)
#     elif sentiment == "['t']":
#         st.markdown('##### You have a *thinking* personalityl')
#         # st.image('images/neutral_gogo.png', width=200)
#     # else:
#     #     st.markdown('##### This tweet is negative')
#     #     st.image('images/cry_gogo.png', width=200)
# # else:
# #     st.markdown('##### *i am waiting......*')

#     lhood = likelihood(text_input)
#     st.write('  ', lhood,"% probabilty this is correct")


################################ GENERATE RANDOM POSTS ################################

st.markdown("---")

st.markdown("## Analyze the posts in the dataset")

df = pd.read_csv('./data/cafe_clean.csv')

rand_post = st.button('Generate Random Post')

if rand_post:
    rint = np.random.randint(0, len(df))

    random_text = df['joined_tokens'].iloc[rint]
    sentiment = prediction(random_text)

    if sentiment == "['f']":
        st.markdown('##### *feeling type*')
    elif sentiment == "['t']":
        st.markdown('##### *thinking type*')

    lhood = likelihood(random_text)
    st.write('  ', lhood,"% Probabilty")

    st.write("RANDOM CLEANED POST:")

    st.write(df['joined_tokens'].iloc[rint])
    
    st.write("RANDOM ORIGINAL POST:")
    st.write(df['posts'].iloc[rint])

################################ PLOT HISTOGRAMS ################################


fig, ax = plt.subplots(figsize=(20,8))
ax.hist(df['post_tokens'].apply(lambda x: len(x)), label='cleaned', alpha=.5, bins=100)
ax.hist(df['posts'].apply(lambda x: len(x.split())), label='pre-cleaned', alpha=.5, bins=100)
ax.legend()
plt.title('Distribution of Post Length \n Clean vs Pre-Cleaned');
st.pyplot(fig)


# mbti_lst = list(set(df['type'].values))
# fig, ax = plt.subplots()
# heights = []
# Xs = []
# for mbti in mbti_lst:
#     heights = heights.append(len(df[df['posts'].str.contains(mbti)].index))
#     Xs = Xs.append(str(mbti))

# ax.bar(Xs[0], heights[0])
# ax.title('Data Leakage \n target within predictors')
# ax.ylabel('counts')
# st.pyplot(fig)

# ds_bg = st.button("Data Science Backound of Model...")

# if ds_bg:
#     st.markdown("""Here are Shuke Company, we created a machine learning model that 
#         could predict the Myers-Briggs Type Indicator for all of the personality types:
#          """)
# load_clf = pickle.load(open('final_clf.pkl', 'rb'))




# color1 = st.color_picker('Pick a color')

# primaryColor = str(color1)

# st.sidebar.header('Weston Shuken')
    


# def prediction(tweeeeeet):
#     pred = str(load_clf.predict([tweeeeeet]))
#     return pred

# if show_sentiment is True:
#     sentiment = prediction(tweet_input)
#     # st.text(sentiment)
#     # st.text(type(sentiment))
#     if sentiment == '[1]':
#         st.markdown('##### This tweet is positive')
#         st.image('images/happy_gogo.png', width=200)
#     elif sentiment == '[0]':
#         st.markdown('##### This tweet is neutral')
#         st.image('images/neutral_gogo.png', width=200)
#     else:
#         st.markdown('##### This tweet is negative')
#         st.image('images/cry_gogo.png', width=200)
# else:
#     st.markdown('##### *i am waiting......*')
#     st.image('images/shiba_gif_200.gif')

###################### SEE MORE INFO ##############################

with st.expander('Open to see more info'):
    st.sidebar.markdown("""
    # Myers-Briggs Type Indicator Prediction
    #### By Weston Shuken

    The purpose of this project is to use machine learning algorithms to precict the personality type of a person given their written text in English. 
    The personality type predictions are based on the Myers-Briggs Type Indicator (MBTI) test as the target variable. 
    The features or predictor variables are comments and posts from userson [PersonalityCafe](https://www.personalitycafe.com/). 
    These posts and comments come from users who have explicitley labeled their MBTI personality on their profile. 

    The Myers-Briggs test is a very popular test that ask users approximately 90 questions about their behavior and assigns the user a type of personality based on this assessment. 
    This test takes around 20-30 for someone to complete. 

    There are 16 different personality types using a combination of 8 overall traits. See below:

        Introversion (I) vs Extroversion (E)
        Intuition (N) vs Sensing (S)
        Thinking (T) vs Feeling (F)
        Judging (J) vs Perceiving (P)


    The page on the right provides a journey to the user, where they can either join a non-technical safari  
    ride throught the project, or they can join a techincal, more extensive, look at how the machine learning models perform. 


    """)

# journey_start = st.button("SEE MORE INFO")

# if journey_start:
#     st.sidebar.markdown("""
#     # Myers-Briggs Type Indicator Prediction
#     #### By Weston Shuken

#     The purpose of this project is to use machine learning algorithms to precict the personality type of a person given their written text in English. 
#     The personality type predictions are based on the Myers-Briggs Type Indicator (MBTI) test as the target variable. 
#     The features or predictor variables are comments and posts from userson [PersonalityCafe](https://www.personalitycafe.com/). 
#     These posts and comments come from users who have explicitley labeled their MBTI personality on their profile. 

#     The Myers-Briggs test is a very popular test that ask users approximately 90 questions about their behavior and assigns the user a type of personality based on this assessment. 
#     This test takes around 20-30 for someone to complete. 

#     There are 16 different personality types using a combination of 8 overall traits. See below:

#         Introversion (I) vs Extroversion (E)
#         Intuition (N) vs Sensing (S)
#         Thinking (T) vs Feeling (F)
#         Judging (J) vs Perceiving (P)


#     The page on the right provides a journey to the user, where they can either join a non-technical safari  
#     ride throught the project, or they can join a techincal, more extensive, look at how the machine learning models perform. 


#     """)