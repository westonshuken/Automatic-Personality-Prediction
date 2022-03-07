###################### IMPORTS ################################
import streamlit as st
import pickle
import pandas as pd
import numpy as np

################################ LOADING AND FUNCTIONS ################################

load_model = pickle.load(open('./models/final_model.pkl', 'rb'))
TFIDF = pickle.load(open('./models/final_tfidf.pkl', 'rb'))
df = pd.read_csv('./data/testing_df.csv')

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



# agree = st.button('Yes')
# disagree = st.button('No')

# if agree:
#     st.write("Very optimistic you are, let's try it")
# elif disagree:
#     st.write("Well, let's give it a shot. What do you say?")

st.markdown("# Welcome to the Personality Predictor")
with st.expander("Try it out..."):
    video_file = open('./videos/water_video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.markdown("### What are your thoughts and feelings about this video")
    text_input = st.text_area('')
    show_sentiment = st.button('Send')

    if show_sentiment:
        sentiment = prediction(text_input)
        lhood = likelihood(text_input)
        
        if sentiment == "['f']":
            st.markdown('#### You have a *feeling* personality')
            html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood}% probabilty this is correct</p>"""
            st.markdown(html_str, unsafe_allow_html=True)
            tfidf_scores = tfidf_top(text_input)
            st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)
            st.markdown("###### According to Myers-Briggs: You believe you can make the best decisions by weighing what people care about and the points-of-view of persons involved in a situation. You are concerned with values and what is the best for the people involved. You like to do whatever will establish or maintain harmony. In your relationships, you appear caring, warm, and tactful.")
        elif sentiment == "['t']":
            st.markdown('#### You have a *thinking* personality')
            html_str = f"""<style>p.a {{font: bold 24px Courier;}}</style><p class="a">{lhood}% probabilty this is correct</p>"""
            st.markdown(html_str, unsafe_allow_html=True)
            tfidf_scores = tfidf_top(text_input)
            st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)
            # st.subheader(lhood, "% probabilty this is correct")
            st.markdown("###### According to Myers-Briggs: When you make a decision, you like to find the basic truth or principle to be applied, regardless of the specific situation involved. You like to analyze pros and cons, and then be consistent and logical in deciding. You try not to be impersonal, so you won't let my personal wishes--or other people's wishes--influence me.")

        # lhood = likelihood(text_input)
        # st.write('  ', lhood,"% probabilty this is correct")
        st.markdown("---")
        st.markdown("# Personality Type Predictor")
        st.markdown("##### Can personality type be predicted based on word choice, text style, and online commenting behavior?")

################################ GENERATE RANDOM POSTS ################################

with st.expander("Generate Random Post from Dataset"):
    rand_post = st.button('Generate')

    if rand_post:
        rint = np.random.randint(0, len(df))

        random_text = df['joined_tokens'].iloc[rint]
        sentiment = prediction(random_text)
        lhood = likelihood(random_text)
        tfidf_scores = tfidf_top(random_text)



        if sentiment == "['f']":
            st.markdown('##### *feeling type*')
            st.write('  ', lhood,"% Probabilty")
            st.write("These terms from your response provide the most weight in determining your personality type \n", tfidf_scores)
        elif sentiment == "['t']":
            st.markdown('##### *thinking type*')
            st.write("These terms from the response provide the most weight in determining the personality type \n", tfidf_scores)

        st.write("ORIGINAL POST:")
        st.write(df['posts'].iloc[rint])
        
        st.write("CLEANED POST:")
        st.write(df['joined_tokens'].iloc[rint])
        



################################ FORM COLLECT DATA ################################
# text_input1 = st.text_area('How are you doing today?')
# btn1 = st.button('Move on')
# if btn1:
#     text_input2 = st.text_area('What is your favorite thing about today?')
#     btn2 = st.button('Next Question')
#     if btn2:
#         text_input3 = st.text_area('What is your least favorite thing about today?')
#     btn3 = st.button('Last Question')
#     if btn3:
#         text_input4 = st.text_area('Anything else on your mind? Write anything you want :-)')
#     show_sentiment = st.button('Blast Off')
#     if show_sentiment:
#         sentiment = prediction(text_input1)
#         if sentiment == "['f']":
#             st.markdown('##### You have a *feeling* personality')
#                 #     st.image('images/happy_gogo.png', width=200)
#         elif sentiment == "['t']":
#             st.markdown('##### You have a *thinking* personalityl')
#                     # st.image('images/neutral_gogo.png', width=200)
#                 # else:
#                 #     st.markdown('##### This tweet is negative')
#                 #     st.image('images/cry_gogo.png', width=200)
#                 # else:
#                 #     st.markdown("###### *don't be shy*")

#         lhood = likelihood(text_input1)
#         st.write('  ', lhood,"% probabilty this is correct")


# st.markdown("# Can personality type be predicted based on word choice, text style, and online commenting behavior?")

# agree = st.button('Yes')
# disagree = st.button('No')


# if agree:
#     st.write('Very optimistic you are')
#     text_input = st.text_area("Type something, let's try it!")
#     btn1 = st.button('Move on')
# elif disagree:
#     st.write("Let's give it a shot")
#     text_input = st.text_area("Type something, anything. How are you doing?")
#     btn1 = st.button('Move on')

#     if btn1:
#         sentiment = prediction(text_input)
#         lhood = likelihood(text_input)
#         if sentiment == "['f']":
#             st.markdown("##### The model believes it is a " + lhood + "% probabilty you have a *feeling* personality")
#                 #     st.image('images/happy_gogo.png', width=200)
#         elif sentiment == "['t']":
#             st.markdown('##### You have a *thinking* personalityl')
#                     # st.image('images/neutral_gogo.png', width=200)
#                 # else:
#                 #     st.markdown('##### This tweet is negative')
#                 #     st.image('images/cry_gogo.png', width=200)
#                 # else:
#                 #     st.markdown("###### *don't be shy*")
#         st.write('The model believes it is a ', lhood,"% change this is correct.")



################################ FORM PREDICTION ################################


# with st.form("Form"):
#     st.write("Would you mind answering a few questions?")
#     text_input1 = st.text_area('How are you doing today?')
#     text_input2 = st.text_area('What is your favorite thing about today?')
#     text_input3 = st.text_area('What is your least favorite thing about today?')
#     text_input4 = st.text_area('Anything else on your mind? Write anything you want :-)')

#     submitted = st.form_submit_button("Submit")
#     total_text = str(text_input1 + text_input2 + text_input3 + text_input4)
#     if submitted and len(total_text) > 100:
#         sentiment = prediction(total_text)
#         lhood = likelihood(total_text)
#         if sentiment == "['f']":
#             st.markdown("##### The model believes it is a " + lhood + "% probabilty you have a *feeling* personality")
#                 #     st.image('images/happy_gogo.png', width=200)
#         elif sentiment == "['t']":
#             st.markdown('##### You have a *thinking* personalityl')
#                     # st.image('images/neutral_gogo.png', width=200)
#                 # else:
#                 #     st.markdown('##### This tweet is negative')
#                 #     st.image('images/cry_gogo.png', width=200)
#                 # else:
#                 #     st.markdown("###### *don't be shy*")
#         st.write('The model believes it is a ', lhood,"% change this is correct.")
#     elif submitted and len(total_text) <= 100:
#         st.write("Can you please write a bit more?")
#         st.text(str(len(total_text)))


################################ USER PREDICTION ################################

# text_input1 = st.text_area('How are you doing today?')
# btn1 = st.button('Move on')
# if btn1:
#     text_input2 = st.text_area('What is your favorite thing about today?')
#     btn2 = st.button('Next Question')
#     if btn2:
#         text_input3 = st.text_area('What is your least favorite thing about today?')
#     btn3 = st.button('Last Question')
#     if btn3:
#         text_input4 = st.text_area('Anything else on your mind? Write anything you want :-)')
#     show_sentiment = st.button('Blast Off')
#     if show_sentiment:
#         sentiment = prediction(text_input1)
#         if sentiment == "['f']":
#             st.markdown('##### You have a *feeling* personality')
#                 #     st.image('images/happy_gogo.png', width=200)
#         elif sentiment == "['t']":
#             st.markdown('##### You have a *thinking* personalityl')
#                     # st.image('images/neutral_gogo.png', width=200)
#                 # else:
#                 #     st.markdown('##### This tweet is negative')
#                 #     st.image('images/cry_gogo.png', width=200)
#                 # else:
#                 #     st.markdown("###### *don't be shy*")

#         lhood = likelihood(text_input1)
#         st.write('  ', lhood,"% probabilty this is correct")


################################ USER ANSWER ################################

# st.markdown('# Are you a *thinking* or *feeling* type of personality?')

# st.write('')
# st.write('')
# st.write('')

# col1, col2 = st.columns(2)
# st.write('')
# st.write('')
# think = col1.button('Thinking')
# feel = col2.button('Feeling')


# if think:
#     st.markdown("When you make a decision, you like to find the basic truth or principle to be applied, regardless of the specific situation involved. You like to analyze pros and cons, and then be consistent and logical in deciding. You try not to be impersonal, so you won't let my personal wishes--or other people's wishes--influence me.")
    
#     tru = st.checkbox("True")
#     ntru = st.checkbox("Not True")
# if tru is True:
#     st.write("Very cool, you really know yourself. This information about the `Feeling` personality comes direcetly from the Myers-Briggs Type Indecator website.")
# if ntru is True:
#     st.write("Fair enough. This comes direcetly from the Myers-Briggs Type Indecator website.")

# if feel:
#     st.markdown("You believe you can make the best decisions by weighing what people care about and the points-of-view of persons involved in a situation. You are concerned with values and what is the best for the people involved. You like to do whatever will establish or maintain harmony. In your relationships, you appear caring, warm, and tactful.")
#     tru = st.checkbox("True")
#     ntru = st.checkbox("Not True")

# if tru is True:
#     st.write("Very cool, you really know yourself. This information about the `Feeling` personality comes direcetly from the Myers-Briggs Type Indecator website.")
# if ntru is True:
#     st.write("Fair enough. This comes direcetly from the Myers-Briggs Type Indecator website.")

# st.markdown("---")

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




################################ PLOT HISTOGRAMS ################################


# fig, ax = plt.subplots(figsize=(20,8))
# ax.hist(df['post_tokens'].apply(lambda x: len(x)), label='cleaned', alpha=.5, bins=100)
# ax.hist(df['posts'].apply(lambda x: len(x.split())), label='pre-cleaned', alpha=.5, bins=100)
# ax.legend()
# plt.title('Distribution of Post Length \n Clean vs Pre-Cleaned');
# st.pyplot(fig)


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