import streamlit as st
import pickle


st.title('Are you a thinking or feeling type of person?')
st.markdown("---")

col1, col2 = st.columns(2)
think = col1.button('Thinking')
feel = col2.button('Feeling')

st.markdown("---")

if think:
    st.markdown("When you make a decision, you like to find the basic truth or principle to be applied, regardless of the specific situation involved. You like to analyze pros and cons, and then be consistent and logical in deciding. You try not to be impersonal, so you won't let my personal wishes--or other people's wishes--influence me.")
    
    tru = st.checkbox("True")
    ntru = st.checkbox("Not True")
if tru is True:
    st.write("Very cool, you really know yourself. This information about the `Feeling` personality comes direcetly from the Myers-Briggs Type Indecator website.")
if ntru is True:
    st.write("Fair enough. This comes direcetly from the Myers-Briggs Type Indecator website.")

if feel:
    st.markdown("You believe you can make the best decisions by weighing what people care about and the points-of-view of persons involved in a situation. You are concerned with values and what is the best for the people involved. You like to do whatever will establish or maintain harmony. In your relationships, you appear caring, warm, and tactful.")
    tru = st.checkbox("True")
    ntru = st.checkbox("Not True")

if tru is True:
    st.write("Very cool, you really know yourself. This information about the `Feeling` personality comes direcetly from the Myers-Briggs Type Indecator website.")
if ntru is True:
    st.write("Fair enough. This comes direcetly from the Myers-Briggs Type Indecator website.")

st.markdown("---")

st.markdown("## Let's use our model to see if we can predict right")

text_input = st.text_area('How are you feeling today?')

show_sentiment = st.button('Send')

# ds_bg = st.button("Data Science Backound of Model...")

# if ds_bg:
#     st.markdown("""Here are Shuke Company, we created a machine learning model that 
#         could predict the Myers-Briggs Type Indicator for all of the personality types:
#          """)
# load_clf = pickle.load(open('final_clf.pkl', 'rb'))




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

