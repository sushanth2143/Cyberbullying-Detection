import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer

# 1. Setup and Preprocessing
st.set_page_config(page_title="SafeGuard NLP", page_icon="🛡️")

stemmer = PorterStemmer()

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once"
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# 2. Expanded Dataset (~320 Samples)

data = [
    # ---- Bullying (1) ----
    ("you are so ugly and pathetic", 1),
    ("i hate you so much go away", 1),
    ("everyone thinks you are a loser", 1),
    ("kill yourself nobody likes you", 1),
    ("you are stupid and worth nothing", 1),
    ("nobody wants to be your friend", 1),
    ("you are such an idiot", 1),
    ("why are you so dumb", 1),
    ("you are completely useless", 1),
    ("go disappear nobody needs you", 1),
    ("you are a failure in life", 1),
    ("stop talking you sound ridiculous", 1),
    ("you embarrass yourself every day", 1),
    ("you are the worst person ever", 1),
    ("everyone laughs at you", 1),
    ("you are a joke", 1),
    ("i can't stand you", 1),
    ("you ruin everything", 1),
    ("nobody cares about you", 1),
    ("you are disgusting", 1),

    # ---- Safe / Neutral (0) ----
    ("the weather is very nice today", 0),
    ("i love your new shoes", 0),
    ("can you help me with this homework", 0),
    ("good luck on your exam tomorrow", 0),
    ("have a great day ahead", 0),
    ("you did a fantastic job", 0),
    ("keep working hard you will succeed", 0),
    ("let us meet tomorrow for discussion", 0),
    ("this project looks amazing", 0),
    ("thank you for your help", 0),
    ("congratulations on your achievement", 0),
    ("i appreciate your effort", 0),
    ("that was a wonderful presentation", 0),
    ("hope you are doing well", 0),
    ("nice to see you today", 0),
    ("best wishes for your future", 0),
    ("let us solve this problem together", 0),
    ("you are very talented", 0),
    ("great teamwork everyone", 0),
    ("keep smiling and stay positive", 0),
] * 8   # 40 base samples × 8 = 320 samples

df = pd.DataFrame(data, columns=['text', 'label'])
df['clean_text'] = df['text'].apply(preprocess)

# 3. Model Training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# 4. Streamlit UI
st.title("🛡️ SafeGuard NLP")
st.markdown("### Societal Problem: Cyberbullying Detection")
st.info("This app uses TF-IDF and Logistic Regression to identify harmful intent.")

user_input = st.text_input("Enter a message to analyze:", placeholder="e.g., Have a great day!")

if st.button("Run Full Pipeline"):
    if user_input:
        cleaned = preprocess(user_input)
        vec_input = vectorizer.transform([cleaned])
        prob = model.predict_proba(vec_input)[0][1]
        is_bullying = prob > 0.5

        if is_bullying:
            st.error(f"**Detected: Cyberbullying** (Score: {prob:.2%})")
        else:
            st.success(f"**Detected: Neutral / Safe** (Score: {prob:.2%})")

        with st.expander("See NLP Pipeline Details"):
            st.write(f"**1. Lowercasing/Cleaning:** {user_input.lower()}")
            st.write(f"**2. Tokenization:** {user_input.lower().split()}")
            st.write(f"**3. Stemmed & Filtered:** {cleaned}")

        with st.expander("Model Performance"):
            st.write(f"**Training Samples:** {len(df)}")
            st.write(f"**Vocabulary Size:** {len(vectorizer.vocabulary_)}")
            st.write("**Model Type:** Logistic Regression + TF-IDF")
    else:
        st.warning("Please enter text first.")
