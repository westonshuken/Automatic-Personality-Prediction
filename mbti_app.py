import streamlit as st
import pickle


st.title('Myers-Briggs Type Predictor')

# load_clf = pickle.load(open('final_clf.pkl', 'rb'))


text_input = st.text_area('How are you feeling today?')

show_sentiment = st.button('Send')

color1 = st.color_picker('Pick a color')

primaryColor = str(color1)

st.sidebar.header('Weston Shuken')
    


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

