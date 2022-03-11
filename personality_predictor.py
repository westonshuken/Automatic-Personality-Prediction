###################### IMPORTS ################################
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import time


################################ LOADING AND FUNCTIONS ################################

load_model = pickle.load(open('./models/final_model.pkl', 'rb'))
TFIDF = pickle.load(open('./models/final_tfidf.pkl', 'rb'))
df = pd.read_csv('./data/testingsample_df.csv')

@st.cache()

def prediction(text):
    pred = str(load_model.predict([text]))
    return pred

def likelihood(text):
    likelihood = round(max(load_model.predict_proba([text])[0])*100, 2)
    return likelihood

def tfidf_top(text, n=10):
    feature_array = np.array(TFIDF.get_feature_names())
    response = TFIDF.transform([text])
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    n = n
    top_n = feature_array[tfidf_sorting][:n]
    return list(top_n)

################################ INTRO / USER PREDICTION ################################
# st.image('./images/header.png', width=100)

st.markdown("# Automatic Personality Predictor")

st.markdown("### Type or paste some text... anything")
text_input = st.text_area('')
show_sentiment = st.button('Predict')


# with st.spinner('Wait for it... '):

if show_sentiment:
    my_bar = st.progress(0)
    sentiment = prediction(text_input)
    lhood = likelihood(text_input)
    # st.markdown('### Our magic sorting hat says...')
    # html_img = '<iframe src="https://giphy.com/embed/9Sc3xiTns7y8w" width="480" height="247" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/harry-potter-mys-hp-9Sc3xiTns7y8w">via GIPHY</a></p>'
    # st.markdown(html_img, unsafe_allow_html=True)
    for percent_complete in range(100):
        time.sleep(.01)
        my_bar.progress(percent_complete + 1)
    if sentiment == "['f']":
        st.markdown('#### You have a *feeling* personality')
        lhood_str = str(lhood)
        html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood_str}% probabilty this is correct</p>"""
        st.markdown(html_str, unsafe_allow_html=True)
        # Bar Graph
        dict = {'Personality': ['FEELING', 'THINKING'], 'Liklihood (%)': [lhood,100-lhood]}
        df_bar = pd.DataFrame(dict, index=[0,1])
        fig = px.bar(df_bar, 
                x = 'Personality',
                y = 'Liklihood (%)',
                color='Personality',
                text_auto=True,
                title = 'Pesonality Type Likelihood Graph')
        st.plotly_chart(fig,  use_container_width=True)
        tfidf_scores = tfidf_top(text_input)
        st.markdown("###### According to Myers-Briggs: You believe you can make the best decisions by weighing what people care about and the points-of-view of persons involved in a situation. You are concerned with values and what is the best for the people involved. You like to do whatever will establish or maintain harmony. In your relationships, you appear caring, warm, and tactful.")
        st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)
    elif sentiment == "['t']":
        st.markdown('#### You have a *thinking* personality')
        lhood_str = str(lhood)
        html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood_str}% probabilty this is correct</p>"""
        st.markdown(html_str, unsafe_allow_html=True)
        # Bar Graph
        dict = {'Personality': ['FEELING', 'THINKING'], 'Liklihood (%)': [100-lhood,lhood]}
        df_bar = pd.DataFrame(dict, index=[0,1])
        fig = px.bar(df_bar, 
                x = 'Personality',
                y = 'Liklihood (%)',
                color='Personality',
                text_auto=True,
                title = 'Pesonality Type Likelihood Graph')
        st.plotly_chart(fig,  use_container_width=True)
        tfidf_scores = tfidf_top(text_input)
        st.markdown("###### According to Myers-Briggs: When you make a decision, you like to find the basic truth or principle to be applied, regardless of the specific situation involved. You like to analyze pros and cons, and then be consistent and logical in deciding. You try not to be impersonal, so you won't let my personal wishes--or other people's wishes--influence me.")
        st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)

################################ GENERATE RANDOM POSTS ################################

st.markdown("### I'd rather generate text")
generate = st.button('Generate')


# with st.spinner('Wait for it...'):

if generate:
    my_bar2 = st.progress(0)

    st.markdown("Generating a \
    collection of 50 user posts from the forum-based website [Personality Cafe](https://www.personalitycafe.com/).")

    st.markdown("Based on the text, the machine learning model will make prediction based of *Thinking* vs. *Feeling* personlity trait.")

    # slider_n = st.slider('Choose how many of the top words to show (by TFIDF)', 1, 30, step=1)
    st.markdown("**Disclaimer**: *posts are from the internet... so they can be offesive.*")
    rint = np.random.randint(0, len(df))

    st.markdown("#### Posts Snippet:")
    post_string = str(df['posts'].iloc[rint])
    st.write(post_string[:150])

    st.write("#### Cleaned Posts Snippet:")
    post_string = str(df['joined_tokens'].iloc[rint])
    st.write(post_string[:100])


    random_text = df['joined_tokens'].iloc[rint]
    actual_type = df['type'].iloc[rint]
    sentiment_g = prediction(random_text)
    lhood_g = likelihood(random_text)
    tfidf_scores_g = tfidf_top(random_text, n=10)

    for percent_complete in range(100):
        time.sleep(.01)
        my_bar2.progress(percent_complete + 1)

    if sentiment_g == "['f']":
        st.markdown('##### *feeling type*')
        st.write("Actual MBTI: ",actual_type.upper())
        html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood_g}% probabilty this is correct</p>"""
        st.markdown(html_str, unsafe_allow_html=True)
        # Bar Graph
        dict = {'Personality': ['FEELING', 'THINKING'], 'Liklihood (%)': [lhood_g,100-lhood_g]}
        df_bar = pd.DataFrame(dict, index=[0,1])
        fig = px.bar(df_bar, 
                x = 'Personality',
                y = 'Liklihood (%)',
                color='Personality',
                text_auto=True,
                title = 'Pesonality Type Likelihood Graph')
        st.plotly_chart(fig,  use_container_width=True)
        tfidf_scores_g = tfidf_top(random_text)
        st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores_g)
    elif sentiment_g == "['t']":
        st.markdown('##### *thinking type*')
        st.write("Actual MBTI: ", actual_type.upper())
        html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood_g}% probabilty this is correct</p>"""
        st.markdown(html_str, unsafe_allow_html=True)
        # Bar Graph
        dict = {'Personality': ['FEELING', 'THINKING'], 'Liklihood (%)': [100-lhood_g,lhood_g]}
        df_bar = pd.DataFrame(dict, index=[0,1])
        fig = px.bar(df_bar, 
                x = 'Personality',
                y = 'Liklihood (%)',
                color='Personality',
                text_auto=True,
                title = 'Pesonality Type Likelihood Graph')
        st.plotly_chart(fig,  use_container_width=True)
        tfidf_scores_g = tfidf_top(random_text)
        st.write("These terms from the response provide the most weight in determining the personality type \n", tfidf_scores_g)


    with st.expander("See Full Posts"):
        st.write("ORIGINAL POST:")
        st.write(df['posts'].iloc[rint])
        
        st.write("CLEANED POST:")
        st.write(df['joined_tokens'].iloc[rint])

################################ MORE INFO ################################

st.markdown("---")

st.markdown("### Learn More")




with st.expander('Project Overview'):
    st.markdown("""
# Automatic Personality Prediction
#### By Weston Shuken
---
Automatic personality detection is the automated forecasting of a personality using human-generated or exchanged contents:

- text
- speech
- videos
- images

The purpose of this project is to use machine learning algorithms to precict the personality type of a person given their written text in English. 
The personality type predictions are based on the Myers-Briggs Type Indicator (MBTI) test as the target variable. 
The features or predictor variables are comments and posts from users on [PersonalityCafe](https://www.personalitycafe.com/). 
These posts and comments come from users who have explicitley labeled their MBTI personality on their profile. 

The Myers-Briggs test is a very popular test that ask users approximately 90 questions about their behavior and assigns the user a type of personality based on this assessment. 
This test takes around 20-30 for someone to complete. 

There are 16 different personality types using a combination of 8 overall traits. See below:

    Introversion (I) vs Extroversion (E)
    Intuition (N) vs Sensing (S)
    Thinking (T) vs Feeling (F)
    Judging (J) vs Perceiving (P)

*If you are unfamilar with the MBTI, please visit [Myers-Briggs Type Indicator](https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/) for more info*

""")


# with st.expander("Why does this matter?"):
#     st.write('This mattesr b')

with st.expander("What kind of model is being used?"):
    st.write('Stochastic Gradient Descent Linear Classifier ([Sk-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)') 

# with st.expander("What is TF-IDF"):
#     st.write('nice')

with st.expander("Still want to see more?"):
    st.write('Check out my [Github](https://github.com/westonshuken/personality-prediction) \
        and feel free to connect with me over [LinkedIn](https://www.linkedin.com/in/westonshuken/).') 

with st.expander("Seeing some issues or have a comment?"):
    st.text_area('Please do let me know!')


