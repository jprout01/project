import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import requests
import streamlit_authenticator as stauth
# Set page configuration
st.set_page_config(layout="wide", page_title="Multilingual Spam Detection", page_icon="🐞")




ps = PorterStemmer()

# Function to translate text to English
def translate_to_english(text, source_lang='auto'):
    # Google Translate API endpoint
    url = 'https://translate.googleapis.com/translate_a/single'

    # Parameters for the translation request
    params = {
        'client': 'gtx',
        'sl': source_lang,
        'tl': 'en',
        'dt': 't',
        'q': text
    }

    try:
        # Sending GET request to the API
        response = requests.get(url, params=params)
        # Extracting translated text from the response
        translated_text = response.json()[0][0][0]
        return translated_text
    except Exception as e:
        st.error("Translation failed. Error: {}".format(e))
        return None

def transform_text(text):
    # Translate text to English
    translated_text = translate_to_english(text)
    
    # Preprocess text
    transformed_text = translated_text.lower()
    transformed_text = nltk.word_tokenize(transformed_text)
    
    y = []
    for i in transformed_text:
        if i.isalnum():
            y.append(i)
    
    transformed_text = y[:]
    y.clear()
    
    for i in transformed_text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    transformed_text = y[:]
    y.clear()
    
    for i in transformed_text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Dynamic Title with CSS 3D Transformation
st.markdown(
    """
    <style>
        /* Title */
        .title {
            font-size: 36px;
            color: #ff9900;
            text-align: center;
            margin-bottom: 30px;
            transition: transform 0.5s;
            transform-style: preserve-3d;
            display: inline-block;
        }
        .title:hover {
            transform: rotateY(10deg);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Multilingual Spam Detection</h1>", unsafe_allow_html=True)

input_sms = st.text_area("Enter the message")

# Translation checkbox
translate_to_english_checkbox = st.checkbox("Translate to English")

# Prediction button
if st.button('Predict'):
    # Check if input text is empty
    if not input_sms.strip():
        st.error("No text has been entered by you.")
    else:
        # Translate to English if selected
        if translate_to_english_checkbox:
            input_sms = translate_to_english(input_sms)

        # Preprocess text
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        prediction_prob = model.predict_proba(vector_input)[0]
        spam_prob = prediction_prob[1] * 100
        not_spam_prob = prediction_prob[0] * 100

        # Display prediction results
        st.subheader("Prediction")
        st.markdown(f"Spam: {spam_prob:.2f}%")
        st.progress(spam_prob / 100)
        st.markdown(f"Not Spam: {not_spam_prob:.2f}%")
        st.progress(not_spam_prob / 100)

        # Additional Features and Enhancements

        # Display Confidence Score
        st.write(f"Confidence Score: {model.predict_proba(vector_input)[0][1]}")

        # Feedback Mechanism
        feedback = st.radio("Is this prediction accurate?", ("Yes", "No"))
        if feedback == "Yes":
            st.write("Thank you for your feedback!")
        else:
            st.write("Please provide more details so we can improve our model.")
