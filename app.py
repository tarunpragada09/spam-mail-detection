import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ------------------- STYLE -------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f0f, #1a0033);
    color: #00ffff;
}

/* Title */
h1 {
    color: #ff00ff;
    text-align: center;
    text-shadow: 0 0 10px #ff00ff;
}

/* Label */
label {
    color: #00ffff !important;
}

/* Text area */
textarea {
    background-color: #000000 !important;
    color: #00ffff !important;
    border: 2px solid #00ffff !important;
    border-radius: 10px;
}

/* Placeholder text */
textarea::placeholder {
    color: #00ffff !important;
    opacity: 0.8;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff00ff, #00ffff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    box-shadow: 0 0 15px #ff00ff;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 25px #00ffff;
    transform: scale(1.05);
}

/* Messages */
.stSuccess {
    color: #00ff00 !important;
    font-size: 18px;
}

.stError {
    color: #ff0033 !important;
    font-size: 18px;
}

.stWarning {
    color: #ffaa00 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------- MODEL -------------------
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

# ------------------- UI -------------------
st.title("📧 Spam Mail Detector")

st.write("Enter your message below to check whether it is spam or not.")

user_input = st.text_area(
    "Email Content",
    height=150,
    placeholder="Enter Your Text Here"
)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚫 This is a Spam Email")
        else:
            st.success("✅ This is Not Spam")

        prob = model.predict_proba(input_data)
        st.write(f"Spam Probability: {prob[0][1]*100:.2f}%")
        