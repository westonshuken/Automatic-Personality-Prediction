###################### IMPORTS ################################
import streamlit as st
import pickle
import pandas as pd
import numpy as np

################################ LOADING AND FUNCTIONS ################################

load_model = pickle.load(open('./models/final_model.pkl', 'rb'))
TFIDF = pickle.load(open('./models/final_tfidf.pkl', 'rb'))
df = pd.read_csv('./data/testingsample_df.csv')

def prediction(text):
    pred = str(load_model.predict([text]))
    return pred

def likelihood(text):
    likelihood = str(round(max(load_model.predict_proba([text])[0])*100, 2))
    return likelihood

def tfidf_top(text, n=5):
    feature_array = np.array(TFIDF.get_feature_names())
    response = TFIDF.transform([text])
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    n = n
    top_n = feature_array[tfidf_sorting][:n]
    return list(top_n)

################################ INTRO / USER PREDICTION ################################

st.markdown("# Automatic Personality Predictor")

st.markdown("### Type or paste some text... anything")
text_input = st.text_area('')
show_sentiment = st.button('Send')

with st.spinner('Wait for it... '):

    if show_sentiment:
        sentiment = prediction(text_input)
        lhood = likelihood(text_input)
        html_img = '<iframe src="https://giphy.com/embed/9Sc3xiTns7y8w" width="480" height="247" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/harry-potter-mys-hp-9Sc3xiTns7y8w">via GIPHY</a></p>'
        st.markdown(html_img, unsafe_allow_html=True)
        st.markdown('### Our magic hat predictor says...')
        
        if sentiment == "['f']":
            st.markdown('#### You have a *feeling* personality')
            html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood}% probabilty this is correct</p>"""
            st.markdown(html_str, unsafe_allow_html=True)
            tfidf_scores = tfidf_top(text_input)
            st.markdown("###### According to Myers-Briggs: You believe you can make the best decisions by weighing what people care about and the points-of-view of persons involved in a situation. You are concerned with values and what is the best for the people involved. You like to do whatever will establish or maintain harmony. In your relationships, you appear caring, warm, and tactful.")
            st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)
        elif sentiment == "['t']":
            st.markdown('#### You have a *thinking* personality')
            html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood}% probabilty this is correct</p>"""
            st.markdown(html_str, unsafe_allow_html=True)
            tfidf_scores = tfidf_top(text_input)
            st.markdown("###### According to Myers-Briggs: When you make a decision, you like to find the basic truth or principle to be applied, regardless of the specific situation involved. You like to analyze pros and cons, and then be consistent and logical in deciding. You try not to be impersonal, so you won't let my personal wishes--or other people's wishes--influence me.")
            st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)

################################ GENERATE RANDOM POSTS ################################
st.markdown("---")
st.markdown("## Generate Text")

st.markdown("The below generator will generate a \
    collection of 50 user posts on a forum-based website [Personality Cafe](personalitycafe.com).")

st.markdown("Based on the text, machine laerning model will make prediction based of thinking vs. feeling personlity trait.")

slider_n = st.slider('Choose how many of the top words to show (by TFIDF)', 1, 30, step=1)
st.markdown("**Disclaimer**: *posts are from the internet... so they can be offesive.*")

rand_post = st.button('Generate')

with st.spinner('Wait for it...'):


    if rand_post:
        rint = np.random.randint(0, len(df))

        random_text = df['joined_tokens'].iloc[rint]
        sentiment = prediction(random_text)
        lhood = likelihood(random_text)
        tfidf_scores = tfidf_top(random_text, n=slider_n)



        if sentiment == "['f']":
            st.markdown('##### *feeling type*')
            html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood}% probabilty this is correct</p>"""
            st.markdown(html_str, unsafe_allow_html=True)
            st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)
        elif sentiment == "['t']":
            st.markdown('##### *thinking type*')
            html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood}% probabilty this is correct</p>"""
            st.markdown(html_str, unsafe_allow_html=True)
            st.write("These terms from the response provide the most weight in determining the personality type \n", tfidf_scores)

        st.markdown("#### Posts Snippet:")
        post_string = str(df['posts'].iloc[rint])
        st.write(post_string[:150])

        st.write("#### Cleaned Posts Snippet:")
        post_string = str(df['joined_tokens'].iloc[rint])
        st.write(post_string[:100])

        with st.expander("See Full Posts"):
            st.write("ORIGINAL POST:")
            st.write(df['posts'].iloc[rint])
            
            st.write("CLEANED POST:")
            st.write(df['joined_tokens'].iloc[rint])

################################ MORE INFO ################################

st.markdown("---")

pressed = st.button('Press if you want to learn more')


if pressed:

    with st.expander('Project Overview'):
        st.markdown("""
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

    
    with st.expander("Why does this matter?"):
        st.write('nice')

    with st.expander("What kind of model is being used?"):
        st.write('nice') 

    with st.expander("What is TF-IDF"):
        st.write('nice')

    with st.expander("Still want to see more?"):
        st.write('Check out my [Github](https://github.com/westonshuken/personality-prediction) \
            and feel free to connect with me over [LinkedIn](https://www.linkedin.com/in/westonshuken/).') 

    with st.expander("Seeing some issues or have a comment?"):
        st.text_area('Please do let me know!')





# """

# MAIN (full journey or test prediction tool)

# Can personality type be predicted based on word choice, text style, and online commenting behavior?
# Yes/No

# USER PREDICTION TYPE BOX --> return personality, likelihood, & TFIDF top words

# There is a popular dataset on Kaggle, called (MBTI) "Myers-Briggs Personality Type Dataset", which 
# includes a large number of people's MBTI type and content written by them on the website Personality Cafe.
# There are dozens of projects which use this dataset to predict MBTI personality traits, however, I was a bit skeptical to believe that this was really achievable. Below are my objective and neutral results on the possibility of this really working. 

# DATA INSIGHT:

# Collections of 50 posts per user of over 8,600 users on Personality Cafe forums. There appear to be no particular subjects or topics overweighting the data, but rather just online chatter.

# BUTTON (generate a random collection of a user's posts)

# DATA CLEANING:
# As you can tell, the posts have a lot of noise and information that is not useful. Using python libraries (pandas, NLTK, textblob), I removed pipe separators, mentions of MBTI type (data leakage), stopwords (common words without much meaning), URLs, symbols, non-ASCII characters, & digits. I also lemmatized the words (e.g. birds --> bird; walked --> walk).

# BUTTON (view cleaned posts)

# SHOW(graphs of 

# ---------------------------------

# SIDEBAR

# The purpose of this project

# Contact information

# Github Repository



# """